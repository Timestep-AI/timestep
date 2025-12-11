"""
Fine-tune SmolVLM2 on Video Captioning and Function Calling

This script fine-tunes SmolVLM2-500M-Video-Instruct on an interleaved dataset containing:
- Video Feedback dataset: Video captioning examples
- Function Calling dataset: Text-only function calling examples with thinking steps

It is designed to run on a Colab A100 for full fine-tuning, but can be squeezed to L4 with QLoRA.

Dependencies:
    pip install accelerate datasets peft bitsandbytes tensorboard pyav num2words
    pip install git+https://github.com/huggingface/transformers.git
    pip install flash-attn --no-build-isolation
"""

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset, interleave_datasets
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import HfApi
from pathlib import Path
import subprocess
import os
import copy
import shutil


# Configuration
USE_LORA = False
USE_QLORA = False

# Model selection
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def load_model_and_processor():
    """
    Load the model and processor.
    
    This script fine-tunes the 500M variant. You can apply QLoRA or LoRA to save memory,
    which loads an adapter to the quantized version of the model.
    If you want to do full fine-tuning, set `USE_LORA` and `USE_QLORA` to False.
    If you want to do LoRA, set `USE_QLORA` to False and `USE_LORA` to True.
    
    The small model should learn more, so we suggest disabling QLoRA or LoRA when fine-tuning it.
    """
    processor = AutoProcessor.from_pretrained(model_id)

    # Load chat template from root and overwrite processor's default
    chat_template_path = Path(__file__).parent / "chat_template.jinja"
    if chat_template_path.exists():
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
        processor.chat_template = chat_template
        processor.tokenizer.chat_template = chat_template
        print(f"✓ Loaded and overwrote chat template from {chat_template_path}")
    else:
        print(f"⚠ Chat template file not found at {chat_template_path}, using default")

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian",
        )
        lora_config.inference_mode = False

        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            device_map="auto",
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to("cuda")

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

    return model, processor


def preprocess_function_calling(sample, processor):
    """
    Preprocess function calling dataset examples.
    
    Adapted from the notebook approach:
    - Handles system messages by merging into first user message with thinking prompt
    - Formats messages using chat template
    - Returns text-only format (no video)
    """
    messages = copy.deepcopy(sample["messages"])
    first_message = messages[0]

    # Instead of adding a system message, we merge the content into the first user message
    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        # Merge system content with the first user message
        if len(messages) > 1:
            messages[1]["content"] = (
                system_message_content
                + " Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n"
                + messages[1]["content"]
            )
        # Remove the system message from the conversation
        messages.pop(0)

    # Convert messages to SmolVLM2 format (text-only, no video)
    formatted_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Convert role names to match SmolVLM2 format
        if role == "human":
            role = "user"
        elif role == "model":
            role = "assistant"
        # Keep tool messages as role="tool" - the chat template will handle them

        # Format content as text-only
        formatted_content = [{"type": "text", "text": content}]
        formatted_messages.append({"role": role, "content": formatted_content})

    return {
        "messages": formatted_messages,
        "dataset_type": "function_calling",  # Mark as function calling example
    }


def load_function_calling_dataset(processor, max_samples=100):
    """
    Load and preprocess the function calling dataset.
    
    Args:
        processor: The processor to use for formatting
        max_samples: Maximum number of samples to use
    
    Returns:
        Preprocessed dataset
    """
    dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"
    print(f"Loading function calling dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name)
    
    # Rename conversations to messages if needed
    if "conversations" in dataset["train"].column_names:
        dataset = dataset.rename_column("conversations", "messages")
    
    # Preprocess the dataset
    def preprocess_fn(sample):
        return preprocess_function_calling(sample, processor)
    
    # Apply preprocessing - remove old messages column since we're replacing it with formatted version
    # Get all column names except messages (we'll replace it)
    old_columns = dataset["train"].column_names
    dataset = dataset.map(preprocess_fn, remove_columns=old_columns)
    
    # Split and select subset
    if "train" in dataset:
        split_ds = dataset["train"].train_test_split(0.1)
        train_ds = split_ds["train"]
    else:
        # If no train split, use the whole dataset
        train_ds = dataset[list(dataset.keys())[0]]
    
    # Use a small subset for fast training
    train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
    
    print(f"Loaded {len(train_ds)} function calling samples")
    
    return train_ds


def load_video_feedback_dataset(max_samples=100):
    """
    Load the video feedback dataset.
    
    Args:
        max_samples: Maximum number of samples to use
    
    Returns:
        Dataset with added dataset_type marker
    """
    ds = load_dataset("TIGER-Lab/VideoFeedback", "real")
    split_ds = ds["train"].train_test_split(test_size=0.5)
    train_ds = split_ds["train"]

    # Use a small subset for fast training (<5 minutes)
    train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    # Add dataset type marker
    def add_type(example):
        example["dataset_type"] = "video_feedback"
        return example
    
    train_ds = train_ds.map(add_type)

    # Clean up
    del split_ds, ds

    # Take a sneak peek
    print(
        f"prompt:  {train_ds[0]['text prompt']}, video: {train_ds[0]['video link']}"
    )
    print(f"Loaded {len(train_ds)} video feedback samples")

    return train_ds


def load_dataset_data(processor, video_samples=100, function_calling_samples=100, val_split=0.2):
    """
    Load both datasets, split into train/validation, and interleave them.
    
    Args:
        processor: The processor to use for preprocessing
        video_samples: Number of video feedback samples to use
        function_calling_samples: Number of function calling samples to use
        val_split: Fraction of data to use for validation (default 0.2 = 20%)
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    # Load video feedback dataset
    video_ds = load_video_feedback_dataset(max_samples=video_samples)
    
    # Load function calling dataset
    fc_ds = load_function_calling_dataset(processor, max_samples=function_calling_samples)
    
    # Split both datasets into train/validation
    video_split = video_ds.train_test_split(test_size=val_split, seed=42)
    fc_split = fc_ds.train_test_split(test_size=val_split, seed=42)
    
    # Interleave train datasets with equal probability
    train_ds = interleave_datasets(
        [video_split["train"], fc_split["train"]],
        probabilities=[0.5, 0.5],  # 50/50 split
        seed=42,
    )
    
    # Interleave validation datasets with equal probability
    val_ds = interleave_datasets(
        [video_split["test"], fc_split["test"]],
        probabilities=[0.5, 0.5],  # 50/50 split
        seed=42,
    )
    
    print(f"Train dataset created with {len(train_ds)} total samples")
    print(f"  - Video feedback: {len(video_split['train'])} samples")
    print(f"  - Function calling: {len(fc_split['train'])} samples")
    print(f"Validation dataset created with {len(val_ds)} total samples")
    print(f"  - Video feedback: {len(video_split['test'])} samples")
    print(f"  - Function calling: {len(fc_split['test'])} samples")
    
    return train_ds, val_ds


def create_collate_fn(processor, model):
    """
    Create data collating function.
    
    Handles both video feedback examples and function calling examples.
    - Video examples: Apply prompt template with videos and captions
    - Function calling examples: Apply prompt template with text-only messages
    """
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        instances = []
        for example in examples:
            dataset_type = example.get("dataset_type", "video_feedback")
            
            if dataset_type == "function_calling":
                # Handle function calling examples (text-only)
                messages = example["messages"]
                
                instance = (
                    processor.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    .to("cuda")
                    .to(model.dtype)
                )
                # Mark as text-only (no pixel_values)
                instance["is_text_only"] = True
            else:
                # Handle video feedback examples
                prompt = example["text prompt"]

                user_content = [{"type": "text", "text": "Caption the video."}]
                user_content.append({"type": "video", "path": example["video link"]})

                messages = [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"{prompt}"}],
                    },
                ]

                instance = (
                    processor.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    .to("cuda")
                    .to(model.dtype)
                )
                instance["is_text_only"] = False
            
            instances.append(instance)

        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100,
        )

        labels[labels == image_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Step 1: figure out maximum frames, height, width across the batch
        # Only consider video examples (not text-only function calling examples)
        pvs = []
        for inst in instances:
            if not inst.get("is_text_only", False) and "pixel_values" in inst:
                pv = inst["pixel_values"]
                if pv is not None:
                    pvs.append(pv.squeeze(0) if pv.dim() > 4 else pv)
        
        if pvs:  # there is at least one non-None pixel_values from video examples
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            # Default values when no video examples in batch
            # SmolVLM typically uses 448x448 for images/videos
            max_h = max_w = 448
            max_frames = 1

        padded_pixel_values_list = []
        for ex in instances:
            is_text_only = ex.get("is_text_only", False)
            pv = ex.get("pixel_values", None)
            
            if is_text_only or pv is None:
                # text-only => fill pixel data with zeros (minimal size)
                shape_pv = (max_frames, 3, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32, device="cuda")
            else:
                if pv.dim() > 4:
                    pv = pv.squeeze(0)
                f, c, h, w = pv.shape
                # Prepare final storage
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device,
                )
                padded_pv[:f, :, :h, :w] = pv
            padded_pixel_values_list.append(padded_pv)

        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

    return collate_fn


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics. Simplified to avoid OOM from storing all predictions.
    
    Note: This function is called during evaluation, but storing all predictions
    can cause OOM. We'll keep it simple and just return empty dict - loss is
    already computed by Trainer automatically.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
    
    Returns:
        Dictionary with metrics (empty for now to save memory)
    """
    # Return empty dict - loss is already tracked by Trainer
    # Computing additional metrics here would require storing all predictions
    # which causes OOM. Loss tracking is sufficient for monitoring.
    return {}


def setup_training(model_id, model, collate_fn, train_ds, val_ds=None, iteration=1, failure_analysis=None):
    """
    Setup training arguments and trainer with adaptive parameters based on iteration and failures.
    
    Some notes:
    - If you use 8-bit QLoRA with the below setup it uses around 16.4 GB VRAM
      (beautiful, fits comfortably inside L4, Colab free tier)
    - We use gradient accumulation to simulate a larger batch size.
    - We also save up on memory from intermediate activations by using gradient checkpointing.
    
    Args:
        model_id: Base model ID
        model: The model to train
        collate_fn: Collate function for batching
        train_ds: Training dataset
        val_ds: Validation dataset (optional, for evaluation during training)
        iteration: Current training iteration (for adaptive step count)
        failure_analysis: Dictionary from analyze_test_failures() with suggested adjustments
    
    Disclaimer:
    The techniques here aren't free lunch. The latter two will add additional compute
    to the training, thus slow down a bit (for reference on two A100s with bsz of 16,
    we were able to train for 2 hrs 43 mins with the gradient accumulation steps of 4,
    disabling it reduced it with 2 hr 35 mins).
    If you want to speed-up, you might play around, reduce to 4-bit precision and have
    a higher batch size. Note that 4-bit might result in model learning less.
    """
    model_name = model_id.split("/")[-1]
    hub_model_id = f"{model_name}-GGUF"
    output_dir = f"data/models/{hub_model_id}"
    
    # Enable gradient checkpointing on the model to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Adaptive training parameters based on iteration and failures
    base_steps = 50
    base_lr = 1e-4
    
    # Increase steps each iteration (cumulative training)
    max_steps = base_steps * iteration
    
    # Adjust based on failure analysis
    if failure_analysis:
        max_steps += failure_analysis.get("suggested_steps_increase", 0)
        lr_adjustment = failure_analysis.get("suggested_lr_adjustment", 0.0)
        learning_rate = max(1e-5, base_lr + lr_adjustment)  # Don't go below 1e-5
    else:
        learning_rate = base_lr
    
    # Cap max steps to prevent excessive training time
    max_steps = min(max_steps, 500)
    
    # Determine evaluation frequency (every 15 steps for frequent evaluation)
    eval_steps = 15
    save_steps = eval_steps if val_ds is not None else max_steps
    
    print(f"  Training parameters: max_steps={max_steps}, learning_rate={learning_rate:.2e}")
    if val_ds is not None:
        print(f"  Evaluation: every {eval_steps} steps")

    training_args = TrainingArguments(
        max_steps=max_steps,
        per_device_train_batch_size=2,  # Reduced for smaller GPUs (increase if you have more memory)
        per_device_eval_batch_size=1,  # Reduced eval batch size to prevent OOM
        gradient_accumulation_steps=2,  # Effective batch size = 2 * 2 = 4
        gradient_checkpointing=True,  # Save memory by trading compute for memory
        warmup_steps=min(5, max_steps // 10),  # Adaptive warmup
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,  # More frequent logging for short training
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=eval_steps if val_ds is not None else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3 if val_ds is not None else 1,  # Keep best 3 checkpoints if evaluating
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
        optim="adamw_torch",  # for 8-bit, use paged_adamw_8bit, else adamw_torch
        bf16=True,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False,
    )

    # Setup callbacks
    callbacks = []
    if val_ds is not None:
        try:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            ))
        except Exception:
            # If EarlyStoppingCallback has issues, continue without it
            print("  Warning: Could not setup EarlyStoppingCallback, continuing without it")

        trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # Don't use compute_metrics to avoid OOM from prediction accumulation
        # Loss is automatically tracked by Trainer
        callbacks=callbacks,
    )

    return trainer


def convert_and_push_gguf(model_path, hub_model_id, quantizations=None):
    """
    Convert model to GGUF format with multiple quantization variants and push to Hugging Face Hub.
    
    Uses the llama.cpp submodule at 3rdparty/llama.cpp.
    First converts to f16 base format, then quantizes to all variants using llama-quantize.
    
    Args:
        model_path: Path to the local model directory
        hub_model_id: Hugging Face Hub model ID to push to (can be just model name or username/model-name)
        quantizations: List of quantization types to generate, or a single string. 
                      If None, defaults to ["Q4_K_M"] for faster iteration.
    """
    api = HfApi()
    
    # Ensure hub_model_id includes username
    if "/" not in hub_model_id:
        user_info = api.whoami()
        username = user_info["name"]
        hub_model_id = f"{username}/{hub_model_id}"
    
    # Create repo if it doesn't exist
    api.create_repo(hub_model_id, exist_ok=True, repo_type="model")
    
    # Use the submodule path
    script_dir = Path(__file__).parent
    llama_cpp_path = script_dir / "3rdparty" / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found at {convert_script}. "
            "Make sure the submodule is initialized: git submodule update --init --recursive"
        )
    
    # Handle single string (backward compatibility) or list
    if isinstance(quantizations, str):
        quantizations = [quantizations]
    
    # Default quantization: Q4_K_M only for faster iteration
    # Note: f16 is the base format, Q4_K_M is quantized from it using llama-quantize
    if quantizations is None:
        quantizations = ["Q4_K_M"]
    
    # Ensure model_path is absolute since we change working directory
    model_path_abs = Path(model_path).resolve()
    if not model_path_abs.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path_abs}")
    
    base_model_name = hub_model_id.split('/')[-1]
    
    # Step 1: Convert to f16 base format first
    print("\n[Step 1/2] Converting model to GGUF f16 base format...")
    base_gguf_file = (llama_cpp_path / f"{base_model_name}-f16.gguf").resolve()
    
    try:
        subprocess.run(
            [
                "python3",
                str(convert_script),
                str(model_path_abs),
                "--outfile",
                str(base_gguf_file),
                "--outtype",
                "f16",
            ],
            check=True,
            cwd=str(llama_cpp_path),
        )
        print(f"✓ Base f16 model created: {base_gguf_file.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create base f16 model: {e}")
    
    # Step 2: Find llama-quantize binary (system-installed or build it)
    quantize_binary = None
    
    # First, try to find system-installed llama-quantize
    system_quantize = shutil.which("llama-quantize") or shutil.which("quantize")
    if system_quantize:
        quantize_binary = Path(system_quantize)
        print(f"Found system-installed quantize tool: {quantize_binary}")
    
    # Try common locations for the quantize binary in the submodule
    if quantize_binary is None:
        possible_paths = [
            llama_cpp_path / "build" / "bin" / "llama-quantize",
            llama_cpp_path / "build" / "bin" / "quantize",
            llama_cpp_path / "bin" / "llama-quantize",
            llama_cpp_path / "bin" / "quantize",
        ]
        
        for path in possible_paths:
            if path.exists():
                quantize_binary = path
                break
    
    # If still not found, try to build it
    if quantize_binary is None:
        print("Building llama-quantize tool...")
        build_dir = llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Try cmake build
        try:
            subprocess.run(
                ["cmake", "-B", str(build_dir), "-S", str(llama_cpp_path)],
                check=True,
                cwd=str(llama_cpp_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", "llama-quantize", "-j"],
                check=True,
                cwd=str(llama_cpp_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            quantize_binary = build_dir / "bin" / "llama-quantize"
            if not quantize_binary.exists():
                # Try alternative name
                quantize_binary = build_dir / "bin" / "quantize"
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # If cmake is not available, provide helpful error
            if isinstance(e, FileNotFoundError) and "cmake" in str(e).lower():
                raise RuntimeError(
                    "llama-quantize tool not found and cmake is not installed.\n"
                    "Please either:\n"
                    "  1. Install cmake: sudo apt-get install cmake (or equivalent for your system)\n"
                    "  2. Build llama-quantize manually: cd 3rdparty/llama.cpp && cmake -B build && cmake --build build --target llama-quantize\n"
                    "  3. Install llama-quantize system-wide and ensure it's in your PATH"
                )
            # Try make build (legacy, but might work)
            try:
                subprocess.run(
                    ["make", "llama-quantize", "-j"],
                    check=True,
                    cwd=str(llama_cpp_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                quantize_binary = llama_cpp_path / "llama-quantize"
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    "Could not find or build llama-quantize.\n"
                    "Please build it manually:\n"
                    "  cd 3rdparty/llama.cpp\n"
                    "  cmake -B build\n"
                    "  cmake --build build --target llama-quantize\n"
                    "\n"
                    "Or install cmake and run this script again."
                )
    
    if not quantize_binary or not quantize_binary.exists():
        raise RuntimeError(f"Quantize binary not found at {quantize_binary}")
    
    print(f"Using quantize tool: {quantize_binary}")
    
    # Step 3: Generate and upload each quantization variant
    # Separate f16 from other quantizations
    other_quants = [q for q in quantizations if q.upper() != "F16"]
    f16_in_list = any(q.upper() == "F16" for q in quantizations)
    
    uploaded_count = 0
    
    # Upload f16 base if requested
    if f16_in_list:
        print(f"\n[Uploading] {base_model_name}-f16.gguf...")
        try:
            api.upload_file(
                path_or_fileobj=str(base_gguf_file),
                path_in_repo=f"{base_model_name}-f16.gguf",
                repo_id=hub_model_id,
            )
            print(f"✓ Successfully uploaded {base_model_name}-f16.gguf")
            uploaded_count += 1
        except Exception as e:
            print(f"✗ Failed to upload f16: {e}")
    
    # Quantize and upload other variants
    for i, quantization in enumerate(other_quants, 1):
        print(f"\n[{i}/{len(other_quants)}] Quantizing to {quantization}...")
        
        # Create filename with quantization suffix
        gguf_filename = f"{base_model_name}-{quantization}.gguf"
        gguf_file = (llama_cpp_path / gguf_filename).resolve()
        
        try:
            # Quantize from f16 base
            subprocess.run(
                [
                    str(quantize_binary),
                    str(base_gguf_file),
                    str(gguf_file),
                    quantization.upper(),
                ],
                check=True,
                cwd=str(llama_cpp_path),
            )
            
            # Upload GGUF file to hub
            print(f"Uploading {gguf_filename} to Hugging Face Hub...")
            api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=gguf_filename,
                repo_id=hub_model_id,
            )
            
            print(f"✓ Successfully uploaded {gguf_filename}")
            uploaded_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to quantize {quantization}: {e}")
            continue
        except Exception as e:
            print(f"✗ Failed to upload {quantization}: {e}")
            continue
        finally:
            # Clean up local quantized GGUF file (keep base f16)
            if gguf_file.exists() and gguf_file != base_gguf_file:
                gguf_file.unlink()
    
    # Clean up base f16 file
    if base_gguf_file.exists():
        base_gguf_file.unlink()
    
    print(f"\n✓ Completed uploading {uploaded_count} GGUF variants to {hub_model_id}")


def create_modelfile(hub_model_id, output_path=None):
    """
    Create a Modelfile for Ollama with SmolVLM2's custom chat template.
    
    The SmolVLM2 chat template format:
    - BOS: <|im_start|>
    - User messages: User: {content}<end_of_utterance>\n
    - Assistant messages: Assistant: {content}<end_of_utterance>\n
    - System messages: {content}\n\n
    - Generation prompt: Assistant:
    
    Args:
        hub_model_id: Hugging Face Hub model ID (e.g., "username/model-name")
        output_path: Optional path to save Modelfile. If None, saves to current directory.
    
    Returns:
        Path to the created Modelfile
    """
    # Ensure hub_model_id includes username if not already present
    api = HfApi()
    if "/" not in hub_model_id:
        user_info = api.whoami()
        username = user_info["name"]
        hub_model_id = f"{username}/{hub_model_id}"
    
    # Extract model name from hub_model_id
    model_name = hub_model_id.split("/")[-1]
    
    # SmolVLM2 chat template in Go template format (for Ollama)
    # Based on LLM_CHAT_TEMPLATE_SMOLVLM from llama.cpp
    # Extended to support tool calls and tool messages
    template_content = """<|im_start|>{{ if .System }}{{ .System }}

{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}User: {{ .Content }}<end_of_utterance>
{{ else if eq .Role "assistant" }}{{ if .ToolCalls }}Assistant: <tool_calls>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{ if not (last .Function.Name) }}, {{ end }}{{ end }}
</tool_calls><end_of_utterance>
{{ else }}Assistant: {{ .Content }}<end_of_utterance>
{{ end }}{{ else if eq .Role "tool" }}Tool: {{ .Content }}<end_of_utterance>
{{ end }}{{ end }}Assistant:"""
    
    modelfile_content = f"""FROM hf.co/{hub_model_id}:Q4_K_M

PARAMETER temperature 0.7
PARAMETER num_ctx 4096

TEMPLATE \"\"\"{template_content}\"\"\"

SYSTEM "You are a helpful assistant."
"""
    
    # Determine output path
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / "Modelfile"
    else:
        output_path = Path(output_path)
    
    # Write Modelfile
    output_path.write_text(modelfile_content)
    print(f"✓ Created Modelfile at {output_path}")
    
    # Optionally upload to Hugging Face Hub
    try:
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo="Modelfile",
            repo_id=hub_model_id,
        )
        print(f"✓ Uploaded Modelfile to {hub_model_id}")
    except Exception as e:
        print(f"⚠ Could not upload Modelfile to hub: {e}")
        print("  Modelfile saved locally and can be used with: ollama create {model_name} -f {output_path}")
    
    return output_path


def pull_model_to_ollama(hub_model_id):
    """
    Pull the latest model from Hugging Face to Ollama.
    
    Args:
        hub_model_id: Hugging Face Hub model ID (e.g., "username/model-name")
    
    Returns:
        True if pull succeeded, False otherwise
    """
    # Ensure hub_model_id includes username if not already present
    api = HfApi()
    if "/" not in hub_model_id:
        user_info = api.whoami()
        username = user_info["name"]
        hub_model_id = f"{username}/{hub_model_id}"
    
    # Model identifier for Ollama
    model_ref = f"hf.co/{hub_model_id}:Q4_K_M"
    
    print(f"\nPulling model to Ollama: {model_ref}")
    
    try:
        result = subprocess.run(
            ["ollama", "pull", model_ref],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Successfully pulled {model_ref} to Ollama")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to pull model: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: 'ollama' command not found. Make sure Ollama is installed and in PATH.")
        return False


def analyze_test_failures(test_output):
    """
    Analyze test output to identify failure patterns and suggest training adjustments.
    
    Args:
        test_output: String containing test output
    
    Returns:
        Dictionary with failure analysis and suggested adjustments
    """
    analysis = {
        "instruction_following_issues": False,
        "response_length_issues": False,
        "function_calling_issues": False,
        "suggested_steps_increase": 0,
        "suggested_lr_adjustment": 0.0,
    }
    
    output_lower = test_output.lower()
    
    # Check for instruction following issues (model ignores specific instructions)
    if any(phrase in output_lower for phrase in [
        "does not contain",
        "does not match",
        "expected",
        "failed: output text",
    ]):
        analysis["instruction_following_issues"] = True
        analysis["suggested_steps_increase"] += 25  # More training needed
        analysis["suggested_lr_adjustment"] = -0.2e-4  # Slightly lower LR for stability
    
    # Check for response length issues (too long/short responses)
    if any(phrase in output_lower for phrase in [
        "too long",
        "too short",
        "length",
        "truncated",
    ]):
        analysis["response_length_issues"] = True
        analysis["suggested_steps_increase"] += 15
    
    # Check for function calling specific issues
    if any(phrase in output_lower for phrase in [
        "function",
        "tool",
        "call",
    ]):
        analysis["function_calling_issues"] = True
        analysis["suggested_steps_increase"] += 20
    
    return analysis


def run_test_suite(project_root=None):
    """
    Run the full test suite (Python and TypeScript) using make test.
    
    Args:
        project_root: Path to project root directory. If None, uses the directory containing main.py.
    
    Returns:
        Tuple of (success: bool, failure_analysis: dict, test_output: str)
    """
    if project_root is None:
        project_root = Path(__file__).parent
    else:
        project_root = Path(project_root)
    
    print(f"\nRunning test suite from {project_root}...")
    
    try:
        result = subprocess.run(
            ["make", "test"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ All tests passed!")
        if result.stdout:
            # Print last few lines of output for context
            lines = result.stdout.strip().split("\n")
            if len(lines) > 10:
                print("\nLast 10 lines of test output:")
                print("\n".join(lines[-10:]))
            else:
                print(result.stdout)
        return True, {}, result.stdout
    except subprocess.CalledProcessError as e:
        print("✗ Tests failed!")
        test_output = (e.stdout or "") + (e.stderr or "")
        
        # Analyze failures
        failure_analysis = analyze_test_failures(test_output)
        
        if e.stdout:
            # Print last few lines of output for context
            lines = e.stdout.strip().split("\n")
            if len(lines) > 20:
                print("\nLast 20 lines of test output:")
                print("\n".join(lines[-20:]))
            else:
                print(e.stdout)
        if e.stderr:
            print("\nError output:")
            print(e.stderr)
        
        # Print failure analysis
        if failure_analysis["instruction_following_issues"]:
            print("\n  → Detected: Instruction following issues")
        if failure_analysis["response_length_issues"]:
            print("  → Detected: Response length issues")
        if failure_analysis["function_calling_issues"]:
            print("  → Detected: Function calling issues")
        
        return False, failure_analysis, test_output
    except FileNotFoundError:
        print("✗ Error: 'make' command not found. Make sure make is installed and in PATH.")
        return False, {}, ""


def test_inference(model, processor):
    """
    Test inference on both video captioning and function calling examples.
    
    Tests:
    1. Video captioning - tests the model's ability to caption videos
    2. Function calling - tests the model's ability to handle function calls with thinking
    """
    print("\n" + "="*80)
    print("TEST 1: Video Captioning")
    print("="*80)
    
    # Test video captioning
    video_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption the video."},
                {
                    "type": "video",
                    "path": "https://huggingface.co/datasets/hexuan21/VideoFeedback-videos-mp4/resolve/main/p/p000304.mp4",
                },
            ],
        }
    ]

    video_inputs = (
        processor.apply_chat_template(
            video_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        .to("cuda")
        .to(model.dtype)
    )

    video_generated_ids = model.generate(**video_inputs, do_sample=False, max_new_tokens=64)
    
    # Extract only the newly generated tokens
    input_length = video_inputs["input_ids"].shape[1]
    video_new_tokens = video_generated_ids[0, input_length:]
    video_generated_text = processor.decode(video_new_tokens, skip_special_tokens=True)

    print("User: Caption the video.")
    print(f"Assistant: {video_generated_text}")
    
    print("\n" + "="*80)
    print("TEST 2: Function Calling")
    print("="*80)
    
    # Test function calling with thinking
    # This simulates a function calling scenario where the model should think before calling a function
    function_calling_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant with access to various tools. "
                        "Also, before making a call to a function take the time to plan the function to take. "
                        "Make that thinking process between <think>{your thoughts}</think>\n\n"
                        "What's the weather like in San Francisco?"
                    ),
                }
            ],
        }
    ]

    fc_inputs = (
        processor.apply_chat_template(
            function_calling_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        .to("cuda")
        .to(model.dtype)
    )

    fc_generated_ids = model.generate(**fc_inputs, do_sample=False, max_new_tokens=128)
    
    # Extract only the newly generated tokens
    input_length = fc_inputs["input_ids"].shape[1]
    fc_new_tokens = fc_generated_ids[0, input_length:]
    fc_generated_text = processor.decode(fc_new_tokens, skip_special_tokens=True)

    print("User: What's the weather like in San Francisco?")
    print(f"Assistant: {fc_generated_text}")
    
    print("\n" + "="*80)
    print("Inference tests completed!")
    print("="*80)


def validate_vision_locally(model, processor):
    """
    Tests vision capabilities to ensure we don't break them during training.
    Uses examples from SmolVLM2 documentation with temperature=0 for consistency.

    Returns: (baseline_match: bool, details: str)
    """
    import torch
    from PIL import Image
    import requests
    from io import BytesIO

    print("\n" + "=" * 80)
    print("LOCAL VISION VALIDATION: Image Description")
    print("=" * 80)

    # Test 1: Simple image description (bee image)
    try:
        print("\n  Testing: Image description (bee)")
        response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
        image = Image.open(BytesIO(response.content))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe this image?"},
                ]
            },
        ]

        # Note: image is embedded in messages content, don't pass separately
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract just the assistant's response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text.strip()

        print(f"    Response: {response[:100]}...")

        # Check if response mentions expected keywords (bee, yellow, flower, etc.)
        keywords = ["bee", "yellow", "flower", "insect", "wing"]
        matched_keywords = [kw for kw in keywords if kw.lower() in response.lower()]

        if len(matched_keywords) >= 2:
            print(f"    ✓ Passed (found keywords: {', '.join(matched_keywords)})")
            return True, f"Vision test passed. Response contained expected keywords: {', '.join(matched_keywords)}"
        else:
            print(f"    ✗ Failed (found keywords: {', '.join(matched_keywords) if matched_keywords else 'none'})")
            return False, f"Vision test failed. Expected bee-related keywords, got: {response[:100]}"

    except Exception as e:
        print(f"    ✗ Error during vision test: {e}")
        return False, f"Vision test error: {str(e)}"
    finally:
        print("=" * 80)


def validate_tool_calling_locally(model, processor):
    """
    Validate tool calling capabilities on the local PyTorch model.

    Tests the model's ability to:
    1. Understand tool calling requests
    2. Generate proper thinking process
    3. Format tool calls correctly

    Returns:
        Tuple of (passed: bool, score: float, details: str)
        - passed: Whether basic validation passed
        - score: Quality score (0.0 to 1.0)
        - details: Description of what passed/failed
    """
    print("\n" + "="*80)
    print("LOCAL VALIDATION: Tool Calling")
    print("="*80)

    test_cases = [
        {
            "name": "Simple weather query",
            "prompt": (
                "You are a helpful assistant with access to various tools. "
                "Also, before making a call to a function take the time to plan the function to take. "
                "Make that thinking process between <think>{your thoughts}</think>\n\n"
                "What's the weather like in San Francisco?"
            ),
            "expected_keywords": ["<think>", "</think>"],
            "max_length": 256,
        },
        {
            "name": "Multi-step reasoning",
            "prompt": (
                "You are a helpful assistant with access to various tools. "
                "Also, before making a call to a function take the time to plan the function to take. "
                "Make that thinking process between <think>{your thoughts}</think>\n\n"
                "I need to book a flight from NYC to LA and then get the weather there."
            ),
            "expected_keywords": ["<think>", "</think>"],
            "max_length": 256,
        },
    ]

    total_score = 0.0
    passed_tests = 0
    failed_details = []

    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": test_case["prompt"]}],
            }
        ]

        inputs = (
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            .to("cuda")
            .to(model.dtype)
        )

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=test_case["max_length"],
            temperature=0.1,
        )

        # Extract only new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0, input_length:]
        generated_text = processor.decode(new_tokens, skip_special_tokens=True)

        print(f"    Response: {generated_text[:100]}...")

        # Score this test case
        case_score = 0.0

        # Check for expected keywords
        keywords_found = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in generated_text.lower())
        keyword_score = keywords_found / len(test_case["expected_keywords"])
        case_score += keyword_score * 0.5  # 50% weight on keywords

        # Check response length (not too short, not too long)
        length_ok = 10 < len(generated_text) < test_case["max_length"] * 1.5
        if length_ok:
            case_score += 0.3  # 30% weight on reasonable length

        # Check for coherence (no repetition)
        words = generated_text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.6:  # At least 60% unique words
                case_score += 0.2  # 20% weight on coherence

        total_score += case_score

        if case_score >= 0.6:  # Pass threshold: 60%
            passed_tests += 1
            print(f"    ✓ Passed (score: {case_score:.2f})")
        else:
            failed_details.append(f"{test_case['name']}: score {case_score:.2f}")
            print(f"    ✗ Failed (score: {case_score:.2f})")
            if keyword_score < 1.0:
                missing = [kw for kw in test_case["expected_keywords"] if kw.lower() not in generated_text.lower()]
                print(f"      Missing keywords: {missing}")

    # Calculate overall metrics
    avg_score = total_score / len(test_cases)
    passed = passed_tests >= len(test_cases) * 0.7  # At least 70% of tests must pass

    print(f"\n  Overall: {passed_tests}/{len(test_cases)} tests passed")
    print(f"  Average score: {avg_score:.2f}")

    if passed:
        details = f"Passed {passed_tests}/{len(test_cases)} tests (avg score: {avg_score:.2f})"
        print(f"  ✓ Local validation PASSED")
    else:
        details = f"Only {passed_tests}/{len(test_cases)} tests passed. Failed: {', '.join(failed_details)}"
        print(f"  ✗ Local validation FAILED")

    print("="*80)

    return passed, avg_score, details


def test_local_ollama_model(model_name, project_root):
    """
    Test a local Ollama model against the test suite.

    Args:
        model_name: Name of the Ollama model to test
        project_root: Path to project root directory

    Returns:
        Tuple of (success: bool, test_output: str)
    """
    print(f"\nTesting local Ollama model: {model_name}")

    # Set environment variable to use local Ollama model
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = model_name

    try:
        result = subprocess.run(
            ["make", "test"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        print("✓ Local Ollama tests passed!")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print("✗ Local Ollama tests failed")
        test_output = (e.stdout or "") + (e.stderr or "")
        # Print last few lines for context
        if e.stdout:
            lines = e.stdout.strip().split("\n")
            if len(lines) > 10:
                print("\nLast 10 lines:")
                print("\n".join(lines[-10:]))
        return False, test_output
    except FileNotFoundError:
        print("✗ Error: 'make' command not found")
        return False, ""


def create_local_ollama_model(gguf_path, model_name, modelfile_path):
    """
    Create a local Ollama model from a GGUF file without pushing to HuggingFace.

    Args:
        gguf_path: Path to local GGUF file
        model_name: Name for the Ollama model
        modelfile_path: Path to Modelfile

    Returns:
        True if successful, False otherwise
    """
    print(f"\nCreating local Ollama model: {model_name}")

    # Create a temporary Modelfile that references the local GGUF
    gguf_path = Path(gguf_path).resolve()
    modelfile_path = Path(modelfile_path)

    # Read the existing Modelfile and replace the FROM line
    modelfile_content = modelfile_path.read_text()

    # Replace the FROM line to use local file
    lines = modelfile_content.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("FROM "):
            new_lines.append(f"FROM {gguf_path}")
        else:
            new_lines.append(line)

    temp_modelfile = modelfile_path.parent / f"Modelfile.local"
    temp_modelfile.write_text("\n".join(new_lines))

    try:
        # Create Ollama model from local GGUF
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(temp_modelfile)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Created local Ollama model: {model_name}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create local Ollama model: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: 'ollama' command not found")
        return False
    finally:
        # Clean up temp file
        if temp_modelfile.exists():
            temp_modelfile.unlink()


def main():
    """Main training function with continuous loop: train → pull → test → repeat until all tests pass."""
    import signal
    import sys

    max_iterations = 10
    iteration = 0
    project_root = Path(__file__).parent
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n⚠ Training loop interrupted by user.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load model and processor (only once, before the loop)
    print("Loading model and processor...")
    model, processor = load_model_and_processor()
    
    # Load interleaved datasets with validation split (only once, before the loop)
    print("Loading datasets...")
    # Use minimal video samples (just to verify interleaving works) and focus on function calling
    train_ds, val_ds = load_dataset_data(processor, video_samples=10, function_calling_samples=1000)
    
    # Create collate function (only once, before the loop)
    collate_fn = create_collate_fn(processor, model)
    
    # Track failure analysis and best scores across iterations
    last_failure_analysis = None
    last_eval_metrics = None
    best_validation_loss = float('inf')
    best_tool_calling_score = 0.0

    # Local Ollama model name
    local_model_name = "smolvlm2-local"

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"TRAINING ITERATION {iteration}/{max_iterations}")
        print(f"{'='*80}\n")
        
        try:
            # Setup training with adaptive parameters (recreate trainer each iteration)
            print(f"[Iteration {iteration}] Setting up training with adaptive parameters...")
            trainer = setup_training(model_id, model, collate_fn, train_ds, val_ds=val_ds, iteration=iteration, failure_analysis=last_failure_analysis)
            hub_model_id = trainer.args.hub_model_id
            
            # Step 1: Train model (resume from checkpoint if available)
            print(f"[Iteration {iteration}] Step 1: Training model...")
            # Find the latest checkpoint that matches current max_steps or start fresh
            checkpoint_path = None
            output_dir = Path(trainer.args.output_dir)
            if output_dir.exists():
                # Look for checkpoints
                checkpoints = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], 
                                   key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
                                   reverse=True)
                # Use the latest checkpoint if it exists and is valid
                if checkpoints:
                    latest_checkpoint = checkpoints[0]
                    trainer_state = latest_checkpoint / "trainer_state.json"
                    if trainer_state.exists():
                        checkpoint_path = str(latest_checkpoint)
                        print(f"  Resuming from checkpoint: {latest_checkpoint.name}")
            
            try:
                if checkpoint_path:
                    trainer.train(resume_from_checkpoint=checkpoint_path)
                else:
                    trainer.train(resume_from_checkpoint=None)
            except (ValueError, FileNotFoundError) as e:
                # Checkpoint invalid or not found, start from scratch
                print(f"  Checkpoint issue ({e}), starting from scratch...")
                trainer.train(resume_from_checkpoint=None)
            print("✓ Training completed")
            
            # Log evaluation results and determine if we should continue
            should_proceed_to_gguf = False
            current_validation_loss = float('inf')

            if val_ds is not None:
                try:
                    eval_results = trainer.evaluate()
                    print(f"\n[Iteration {iteration}] Validation metrics:")
                    for key, value in eval_results.items():
                        if key.startswith("eval_"):
                            metric_name = key.replace("eval_", "")
                            print(f"  {metric_name}: {value:.4f}")

                    current_validation_loss = eval_results.get("eval_loss", float('inf'))

                    # Compare with best validation loss
                    if current_validation_loss < best_validation_loss:
                        improvement = best_validation_loss - current_validation_loss
                        print(f"  ✓ New best validation loss! Improved by {improvement:.4f}")
                        best_validation_loss = current_validation_loss
                        should_proceed_to_gguf = True
                    else:
                        degradation = current_validation_loss - best_validation_loss
                        print(f"  ✗ Validation loss degraded by {degradation:.4f} (best: {best_validation_loss:.4f})")
                        print(f"  ⚠ Skipping GGUF conversion - no improvement")

                    last_eval_metrics = eval_results
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  ⚠ Evaluation skipped due to OOM: {e}")
                    print("  Training completed successfully, but evaluation could not run.")
                    should_proceed_to_gguf = True  # Proceed if we can't evaluate
                except Exception as e:
                    print(f"  ⚠ Evaluation failed: {e}")
                    print("  Training completed successfully, but evaluation could not run.")
                    should_proceed_to_gguf = True  # Proceed if we can't evaluate
            else:
                # No validation dataset, always proceed
                should_proceed_to_gguf = True
            
            # Step 2: Save model and processor
            print(f"[Iteration {iteration}] Step 2: Saving model and processor...")
            trainer.save_model()
            processor.save_pretrained(trainer.args.output_dir)
            model_path = Path(trainer.args.output_dir).resolve()
            print("✓ Model saved")

            # Step 3a: Local PyTorch validation for vision (ensure we don't break it)
            print(f"[Iteration {iteration}] Step 3a: Validating vision locally (PyTorch)...")
            vision_passed, vision_details = validate_vision_locally(model, processor)
            if not vision_passed:
                print(f"  ⚠ WARNING: Vision validation failed! {vision_details}")
                print(f"  Continuing anyway, but vision capabilities may be degraded.")

            # Step 3b: Local PyTorch validation for tool calling
            print(f"[Iteration {iteration}] Step 3b: Validating tool calling locally (PyTorch)...")
            local_passed, tool_calling_score, details = validate_tool_calling_locally(model, processor)

            if tool_calling_score > best_tool_calling_score:
                improvement = tool_calling_score - best_tool_calling_score
                print(f"  ✓ New best tool calling score! Improved by {improvement:.2f}")
                best_tool_calling_score = tool_calling_score
            else:
                degradation = best_tool_calling_score - tool_calling_score
                print(f"  ✗ Tool calling score degraded by {degradation:.2f} (best: {best_tool_calling_score:.2f})")

            # Decide whether to proceed based on validation loss AND tool calling score
            if should_proceed_to_gguf and local_passed:
                print(f"\n  ✓ Model improved! Proceeding with GGUF conversion and testing...")
            else:
                reasons = []
                if not should_proceed_to_gguf:
                    reasons.append("validation loss did not improve")
                if not local_passed:
                    reasons.append("local tool calling tests failed")
                print(f"\n  ✗ Skipping GGUF conversion: {' and '.join(reasons)}")
                print(f"  Continuing to iteration {iteration + 1}...")
                continue

            # Step 4: Convert to GGUF locally (Q4_K_M only, don't push yet)
            print(f"[Iteration {iteration}] Step 4: Converting to GGUF (local only)...")
            script_dir = Path(__file__).parent
            llama_cpp_path = script_dir / "3rdparty" / "llama.cpp"
            base_model_name = hub_model_id.split('/')[-1]
            local_gguf_file = llama_cpp_path / f"{base_model_name}-Q4_K_M.gguf"

            # Convert to GGUF locally (copied logic from convert_and_push_gguf but without pushing)
            try:
                convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
                if not convert_script.exists():
                    raise FileNotFoundError(
                        f"llama.cpp convert script not found at {convert_script}. "
                        "Make sure the submodule is initialized: git submodule update --init --recursive"
                    )

                model_path_abs = model_path.resolve()
                base_gguf_file = (llama_cpp_path / f"{base_model_name}-f16.gguf").resolve()

                # Convert to f16 base
                print("  Converting to f16...")
                subprocess.run(
                    [
                        "python3",
                        str(convert_script),
                        str(model_path_abs),
                        "--outfile",
                        str(base_gguf_file),
                        "--outtype",
                        "f16",
                    ],
                    check=True,
                    cwd=str(llama_cpp_path),
                )

                # Find quantize binary
                quantize_binary = shutil.which("llama-quantize") or shutil.which("quantize")
                if not quantize_binary:
                    # Try submodule locations
                    for path in [
                        llama_cpp_path / "build" / "bin" / "llama-quantize",
                        llama_cpp_path / "build" / "bin" / "quantize",
                    ]:
                        if path.exists():
                            quantize_binary = str(path)
                            break

                if not quantize_binary:
                    raise FileNotFoundError("llama-quantize not found. Please build it or install it.")

                # Quantize to Q4_K_M
                print("  Quantizing to Q4_K_M...")
                subprocess.run(
                    [
                        quantize_binary,
                        str(base_gguf_file),
                        str(local_gguf_file),
                        "Q4_K_M",
                    ],
                    check=True,
                    cwd=str(llama_cpp_path),
                )

                # Clean up f16 file
                if base_gguf_file.exists():
                    base_gguf_file.unlink()

                print(f"✓ Local GGUF created: {local_gguf_file}")

            except Exception as e:
                print(f"✗ Failed to create local GGUF: {e}")
                print("  Continuing to next iteration...")
                continue

            # Step 5: Create local Ollama model and test
            print(f"[Iteration {iteration}] Step 5: Creating local Ollama model...")
            modelfile_path = script_dir / "Modelfile"

            # Create Modelfile if it doesn't exist
            if not modelfile_path.exists():
                create_modelfile(hub_model_id, output_path=modelfile_path)

            if not create_local_ollama_model(local_gguf_file, local_model_name, modelfile_path):
                print("✗ Failed to create local Ollama model. Cannot proceed with tests.")
                print("  Continuing to next iteration...")
                continue

            # Step 6: Test local Ollama model
            print(f"[Iteration {iteration}] Step 6: Testing local Ollama model...")
            all_tests_passed, test_output = test_local_ollama_model(local_model_name, project_root)

            if not all_tests_passed:
                # Analyze failures for next iteration
                failure_analysis = analyze_test_failures(test_output)
                last_failure_analysis = failure_analysis
                print(f"\n  ✗ Local Ollama tests failed")
                print(f"  Skipping HuggingFace push. Continuing to iteration {iteration + 1}...")
                continue

            # Step 7: Tests passed! Push to HuggingFace
            print(f"\n{'='*80}")
            print(f"✓ All local tests passed! Pushing to HuggingFace...")
            print(f"{'='*80}\n")

            print(f"[Iteration {iteration}] Step 7: Pushing GGUF to HuggingFace Hub...")
            try:
                api = HfApi()
                if "/" not in hub_model_id:
                    user_info = api.whoami()
                    username = user_info["name"]
                    hub_model_id_full = f"{username}/{hub_model_id}"
                else:
                    hub_model_id_full = hub_model_id

                api.create_repo(hub_model_id_full, exist_ok=True, repo_type="model")

                # Upload the Q4_K_M GGUF file
                api.upload_file(
                    path_or_fileobj=str(local_gguf_file),
                    path_in_repo=f"{base_model_name}-Q4_K_M.gguf",
                    repo_id=hub_model_id_full,
                )
                print(f"✓ Uploaded {base_model_name}-Q4_K_M.gguf to {hub_model_id_full}")

                # Upload Modelfile
                api.upload_file(
                    path_or_fileobj=str(modelfile_path),
                    path_in_repo="Modelfile",
                    repo_id=hub_model_id_full,
                )
                print(f"✓ Uploaded Modelfile to {hub_model_id_full}")

            except Exception as e:
                print(f"✗ Failed to push to HuggingFace: {e}")
                print("  Model works locally but not pushed to HF. Continuing...")

            # Clean up local GGUF file
            if local_gguf_file.exists():
                local_gguf_file.unlink()
                print(f"✓ Cleaned up local GGUF file")

            # Step 8: Success - all tests passed!
            print(f"\n{'='*80}")
            print(f"✓ SUCCESS: All tests passed after {iteration} iteration(s)!")
            print(f"✓ Model pushed to HuggingFace: {hub_model_id_full}")
            print(f"{'='*80}\n")
            break
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training loop interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            if iteration < max_iterations:
                print(f"Continuing to iteration {iteration + 1}...\n")
            else:
                print(f"Reached maximum iterations ({max_iterations}).\n")
    
    if iteration >= max_iterations:
        print(f"\n⚠ Reached max iterations ({max_iterations}) without all tests passing.")
        print("  Consider:")
        print("  - Increasing max_iterations")
        print("  - Adjusting training parameters (learning rate, steps, etc.)")
        print("  - Reviewing test failures to identify specific issues")
    
    # Test inference on final model
    print("\n" + "="*80)
    print("Final inference test")
    print("="*80)
    test_inference(model, processor)


if __name__ == "__main__":
    main()
