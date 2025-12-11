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
        elif role == "tool":
            role = "assistant"  # Treat tool responses as assistant messages
        
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


def load_dataset_data(processor, video_samples=100, function_calling_samples=100):
    """
    Load both datasets and interleave them.
    
    Args:
        processor: The processor to use for preprocessing
        video_samples: Number of video feedback samples to use
        function_calling_samples: Number of function calling samples to use
    
    Returns:
        Interleaved dataset
    """
    # Load video feedback dataset
    video_ds = load_video_feedback_dataset(max_samples=video_samples)
    
    # Load function calling dataset
    fc_ds = load_function_calling_dataset(processor, max_samples=function_calling_samples)
    
    # Interleave datasets with equal probability
    interleaved_ds = interleave_datasets(
        [video_ds, fc_ds],
        probabilities=[0.5, 0.5],  # 50/50 split
        seed=42,
    )
    
    print(f"Interleaved dataset created with {len(interleaved_ds)} total samples")
    print(f"  - Video feedback: {len(video_ds)} samples")
    print(f"  - Function calling: {len(fc_ds)} samples")
    
    return interleaved_ds


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


def setup_training(model_id, model, collate_fn, train_ds):
    """
    Setup training arguments and trainer.
    
    Some notes:
    - If you use 8-bit QLoRA with the below setup it uses around 16.4 GB VRAM
      (beautiful, fits comfortably inside L4, Colab free tier)
    - We use gradient accumulation to simulate a larger batch size.
    - We also save up on memory from intermediate activations by using gradient checkpointing.
    
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

    training_args = TrainingArguments(
        max_steps=50,  # Limit to 50 steps for fast training (<5 minutes)
        per_device_train_batch_size=2,  # Reduced for smaller GPUs (increase if you have more memory)
        gradient_accumulation_steps=2,  # Effective batch size = 2 * 2 = 4
        gradient_checkpointing=True,  # Save memory by trading compute for memory
        warmup_steps=5,  # Reduced warmup for faster training
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,  # More frequent logging for short training
        save_strategy="steps",
        save_steps=50,  # Save at the end
        save_total_limit=1,
        optim="adamw_torch",  # for 8-bit, use paged_adamw_8bit, else adamw_torch
        bf16=True,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
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
                      If None, generates all common variants.
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
    
    # Default quantization variants (similar to bartowski/Llama-3.2-1B-Instruct-GGUF)
    # Note: f16 is the base format, others are quantized from it using llama-quantize
    if quantizations is None:
        quantizations = [
            "IQ3_M",
            "IQ4_XS",
            "Q3_K_L",
            "Q3_K_XL",  # May not exist in all versions, will fail gracefully
            "Q4_0",
            "Q4_K_S",
            "Q4_K_M",
            "Q4_K_L",
            "Q5_K_S",
            "Q5_K_M",
            "Q5_K_L",
            "Q6_K",
            "Q6_K_L",  # May not exist in all versions, will fail gracefully
            "Q8_0",
            "f16",
        ]
    
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


def main():
    """Main training function."""
    # Load model and processor
    model, processor = load_model_and_processor()

    # Load interleaved dataset (video feedback + function calling)
    train_ds = load_dataset_data(processor, video_samples=100, function_calling_samples=100)

    # Create collate function
    collate_fn = create_collate_fn(processor, model)

    # Setup training
    trainer = setup_training(model_id, model, collate_fn, train_ds)

    # Train (will automatically resume from latest checkpoint if available)
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        # No checkpoint found, start from scratch
        trainer.train(resume_from_checkpoint=None)

    # Push to hub as GGUF
    print("Converting and pushing model as GGUF to hub...")
    # First save the model and processor locally
    trainer.save_model()
    processor.save_pretrained(trainer.args.output_dir)
    model_path = Path(trainer.args.output_dir).resolve()
    hub_model_id = trainer.args.hub_model_id
    
    # Convert to GGUF and push
    convert_and_push_gguf(str(model_path), hub_model_id)

    # Test inference
    print("Testing inference...")
    test_inference(model, processor)


if __name__ == "__main__":
    main()
