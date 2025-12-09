#!/usr/bin/env python3
"""
Fine-tune SmolVLM2-500M-Video-Instruct for function calling using unsloth.

This script:
1. Loads the base model and function calling dataset
2. Fine-tunes with LoRA using unsloth
3. Exports to GGUF format with Q4_K_M quantization
4. Publishes to HuggingFace for direct Ollama pulls
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login

# Import unsloth with FastVisionModel for vision-language models
from unsloth import FastVisionModel
from trl import SFTConfig, SFTTrainer

# Model configuration
BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./data/smolvlm2-function-calling"


def load_function_calling_dataset():
    """Load function calling dataset."""
    print("Loading function calling dataset...")
    try:
        # Load Hermes function calling dataset with thinking
        dataset = load_dataset(
            "Jofthomas/hermes-function-calling-thinking-V1",
            split="train"
        )
        print(f"Loaded dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        print("Creating a minimal example dataset for testing...")
        # Create a minimal example dataset for function calling
        from datasets import Dataset
        example_data = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in San Francisco?"
                    },
                    {
                        "role": "assistant",
                        "content": "<think>I need to call the get_weather function to retrieve the current weather in San Francisco.</think>\n<tool_call>{'name': 'get_weather', 'arguments': {'city': 'San Francisco'}}</tool_call>"
                    },
                    {
                        "role": "tool",
                        "content": "<tool_response>72°F and sunny</tool_response>"
                    },
                    {
                        "role": "assistant",
                        "content": "The weather in San Francisco is 72°F and sunny."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Calculate 25 * 17"
                    },
                    {
                        "role": "assistant",
                        "content": "<think>I should use the calculator function to compute 25 multiplied by 17.</think>\n<tool_call>{'name': 'calculator', 'arguments': {'expression': '25 * 17'}}</tool_call>"
                    },
                    {
                        "role": "tool",
                        "content": "<tool_response>425</tool_response>"
                    },
                    {
                        "role": "assistant",
                        "content": "25 * 17 = 425"
                    }
                ]
            }
        ]
        return Dataset.from_list(example_data)


def create_text_only_collator(tokenizer):
    """
    Create a custom data collator for text-only training on vision models.
    This passes images=None to the tokenizer to train without visual inputs.
    Based on: https://github.com/unslothai/unsloth/issues/1590
    """
    def collate_fn(examples):
        # Extract messages from examples
        processed_examples = [example['messages'] for example in examples]

        # Apply chat template to convert to text
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            for messages in processed_examples
        ]

        # Tokenize with images=None for text-only training
        batch = tokenizer(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

        # Create labels (copy of input_ids with padding masked)
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    return collate_fn


def train_model(args):
    """Phase 1: Train the model with LoRA fine-tuning."""
    print("=" * 80)
    print("PHASE 1: TRAINING")
    print("=" * 80)

    if args.skip_training:
        print("Skipping training phase (--skip-training flag set)")
        return None, None

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the vision model using FastVisionModel
    print(f"Loading model: {BASE_MODEL}")
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )

    # Apply LoRA - finetune only language layers for function calling
    # Keep vision layers frozen to preserve multimodal capabilities
    print("Applying LoRA adapters to language layers...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Keep vision frozen
        finetune_language_layers=True,  # Train language for function calling
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,  # LoRA rank
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Load function calling dataset
    dataset = load_function_calling_dataset()

    # Handle different dataset formats
    if "conversations" in dataset.column_names:
        dataset = dataset.rename_column("conversations", "messages")

    # Split dataset if needed
    if hasattr(dataset, "train_test_split"):
        split = dataset.train_test_split(test_size=0.1)
        train_dataset = split["train"]
    else:
        train_dataset = dataset

    # Create custom data collator for text-only training
    data_collator = create_text_only_collator(tokenizer)

    # Training configuration using SFTConfig
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=args.max_steps if args.max_steps else 100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        seed=3407,
        save_steps=10,
        save_total_limit=2,
        # Important settings for text-only vision model training
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Create trainer with custom data collator
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")
    return model, tokenizer


def export_to_gguf_and_publish(args, model=None, tokenizer=None):
    """Phase 2: Export to GGUF and publish to HuggingFace."""
    print("=" * 80)
    print("PHASE 2: GGUF CONVERSION & PUBLISHING")
    print("=" * 80)

    if args.skip_gguf:
        print("Skipping GGUF conversion and publishing (--skip-gguf flag set)")
        return

    # Load model if not provided
    if model is None or tokenizer is None:
        print(f"Loading trained model from {OUTPUT_DIR}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            OUTPUT_DIR,
            load_in_4bit=False,  # Load full model for export
            use_gradient_checkpointing="unsloth"
        )

    # Export to GGUF
    print(f"Exporting model to GGUF format with {args.quantization} quantization...")
    gguf_repo = args.model_name

    # Login to HuggingFace
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not provided. Set --hf-token or HF_TOKEN environment variable.")
        return

    login(token=hf_token)

    # Use unsloth's push_to_hub_gguf method
    try:
        print(f"Pushing to HuggingFace: {gguf_repo}")
        model.push_to_hub_gguf(
            gguf_repo,
            tokenizer,
            quantization_method=args.quantization,
            token=hf_token,
        )
        print(f"Model published to https://huggingface.co/{gguf_repo}")
        print(f"You can now use it with: ollama run hf.co/{gguf_repo}")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Trying to save locally first...")

        # Fallback: save locally
        gguf_output_dir = f"{OUTPUT_DIR}_gguf"
        model.save_pretrained_gguf(
            gguf_output_dir,
            tokenizer,
            quantization_method=args.quantization,
        )
        print(f"GGUF model saved to {gguf_output_dir}")

        # Then try to upload
        api = HfApi()
        try:
            api.create_repo(
                repo_id=gguf_repo,
                repo_type="model",
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=gguf_output_dir,
                repo_id=gguf_repo,
                repo_type="model",
            )
            print(f"Model uploaded to https://huggingface.co/{gguf_repo}")
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM2 for function calling")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mjschock/SmolVLM2-500M-Function-Calling-GGUF",
        help="HuggingFace model name for publishing",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="GGUF quantization level (e.g., q4_k_m, q8_0, f16)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion and publishing",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Phase 1: Training
    model, tokenizer = train_model(args)
    
    # Phase 2: Export and Publish
    export_to_gguf_and_publish(args, model, tokenizer)
    
    print("=" * 80)
    print("ALL PHASES COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

