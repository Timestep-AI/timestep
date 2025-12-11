"""
Fine-tune SmolVLM2 on Video Captioning

This script fine-tunes SmolVLM2-500M-Video-Instruct on Video Feedback dataset.
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
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import HfApi
from pathlib import Path
import subprocess
import os


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


def load_dataset_data():
    """
    Load the dataset and preprocess it.
    
    We will load a dataset that contains generated videos and their super short captions
    of 4k examples. We are loading small chunk of it for training and smaller one for test.
    """
    ds = load_dataset("TIGER-Lab/VideoFeedback", "real")
    split_ds = ds["train"].train_test_split(test_size=0.5)
    train_ds = split_ds["train"]

    # Use a small subset for fast training (<5 minutes)
    train_ds = train_ds.select(range(min(100, len(train_ds))))

    # Clean up
    del split_ds, ds

    # Take a sneak peek
    print(
        f"prompt:  {train_ds[0]['text prompt']}, video: {train_ds[0]['video link']}"
    )
    print(f"Using {len(train_ds)} samples for training")

    return train_ds


def create_collate_fn(processor, model):
    """
    Create data collating function.
    
    We will apply prompt template to have videos and captions together so model can learn
    to caption. Then we pass the formatted prompts and videos to the processor which processes both.
    """
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        instances = []
        for example in examples:
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
        pvs = [
            inst["pixel_values"].squeeze(0)
            for inst in instances
            if "pixel_values" in inst
        ]
        if pvs:  # there is at least one non-None pixel_values
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            max_h = max_w = processor.video_size["longest_edge"]
            max_frames = 1

        padded_pixel_values_list = []
        for ex in instances:
            pv = ex.get("pixel_values", None).squeeze(0)

            if pv is None:
                # text-only => fill pixel data + mask with zeros
                shape_pv = (max_frames, 3, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
            else:
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

    training_args = TrainingArguments(
        max_steps=50,  # Limit to 50 steps for fast training (<5 minutes)
        per_device_train_batch_size=4,  # Increased batch size for faster training
        gradient_accumulation_steps=1,
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


def convert_and_push_gguf(model_path, hub_model_id, quantization="q8_0"):
    """
    Convert model to GGUF format and push to Hugging Face Hub.
    
    Uses the llama.cpp submodule at 3rdparty/llama.cpp.
    
    Args:
        model_path: Path to the local model directory
        hub_model_id: Hugging Face Hub model ID to push to (can be just model name or username/model-name)
        quantization: Quantization type (e.g., "q8_0", "q4_0", "f16", "f32", "bf16")
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
    
    # Convert to GGUF
    gguf_filename = f"{hub_model_id.split('/')[-1]}.gguf"
    # Use absolute path for the output file
    gguf_file = (llama_cpp_path / gguf_filename).resolve()
    print(f"Converting model to GGUF format ({quantization})...")
    
    # Ensure model_path is absolute since we change working directory
    model_path_abs = Path(model_path).resolve()
    if not model_path_abs.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path_abs}")
    
    subprocess.run(
        [
            "python3",
            str(convert_script),
            str(model_path_abs),
            "--outfile",
            str(gguf_file),
            "--outtype",
            quantization,
        ],
        check=True,
        cwd=str(llama_cpp_path),  # Run from llama.cpp directory for proper imports
    )
    
    # Upload GGUF file to hub
    print(f"Uploading {gguf_filename} to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj=str(gguf_file),
        path_in_repo=gguf_filename,
        repo_id=hub_model_id,
    )
    
    # Clean up local GGUF file
    if gguf_file.exists():
        gguf_file.unlink()
    
    print(f"Successfully pushed GGUF model to {hub_model_id}")


def test_inference(model, processor):
    """
    Test inference on a sample video.
    
    The test example is a video of a woman walking by, you can download and check from:
    https://huggingface.co/datasets/hexuan21/VideoFeedback-videos-mp4/blob/main/p/p000304.mp4
    """
    messages = [
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

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])


def main():
    """Main training function."""
    # Load model and processor
    model, processor = load_model_and_processor()

    # Load dataset
    train_ds = load_dataset_data()

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
