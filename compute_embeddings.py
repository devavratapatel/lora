#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import hashlib

import pandas as pd
import torch
from transformers import T5EncoderModel

from diffusers import StableDiffusion3Pipeline


PROMPT = "a photo of sks dog"
MAX_SEQ_LENGTH = 77
LOCAL_DATA_DIR = "dataset"
OUTPUT_PATH = "sample_embeddings.parquet"


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def generate_image_hash(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return hashlib.sha256(img_data).hexdigest()


def load_sd3_pipeline():
    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(id, subfolder="text_encoder_3", load_in_8bit=True, device_map="auto")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        id, text_encoder_3=text_encoder, transformer=None, vae=None, device_map="balanced"
    )
    return pipeline


@torch.no_grad()
def compute_embeddings(pipeline, prompt, max_sequence_length):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, max_sequence_length=max_sequence_length)

    print(
        f"{prompt_embeds.shape=}, {negative_prompt_embeds.shape=}, {pooled_prompt_embeds.shape=}, {negative_pooled_prompt_embeds.shape}"
    )

    max_memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


import os
from tqdm import tqdm # Import tqdm for a progress bar

def run(args):
    pipeline = load_sd3_pipeline()
    
    # We will compute embeddings inside the loop now
    # image_paths = glob.glob(f"{args.local_data_dir}/*.png") # You can use any image format
    image_paths = [p for p in glob.glob(f"{args.local_data_dir}/*") if p.endswith((".png", ".jpg", ".jpeg", ".webp"))]
    
    data = []
    
    print(f"Found {len(image_paths)} images. Processing captions and computing embeddings...")
    
    # Use tqdm for a nice progress bar
    for image_path in tqdm(image_paths):
        # 1. Generate the corresponding caption file path
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        
        # 2. Read the unique caption from the text file
        try:
            with open(caption_path, "r") as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"WARNING: Caption file not found for {image_path}. Skipping.")
            continue
            
        # 3. Compute embeddings FOR THIS SPECIFIC CAPTION
        prompt_embeds, _, pooled_prompt_embeds, _ = compute_embeddings(
            pipeline, prompt, args.max_sequence_length
        )
        
        # 4. Generate the image hash and append the unique data
        img_hash = generate_image_hash(image_path)
        data.append(
            # We are only adding the positive embeddings, as the training script uses them
            (img_hash, prompt_embeds, pooled_prompt_embeds)
        )

    # Create a DataFrame
    # Note the column names have changed to match what we are appending
    embedding_cols = [
        "prompt_embeds",
        "pooled_prompt_embeds",
    ]
    df = pd.DataFrame(
        data,
        columns=["image_hash"] + embedding_cols,
    )

    # Convert embedding lists to arrays (for proper storage in parquet)
    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: x.cpu().numpy().flatten().tolist())

    # Save the dataframe to a parquet file
    df.to_parquet(args.output_path)
    print(f"Data with DYNAMIC embeddings successfully serialized to {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=PROMPT, help="The instance prompt.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length to use for computing the embeddings. The more the higher computational costs.",
    )
    parser.add_argument(
        "--local_data_dir", type=str, default=LOCAL_DATA_DIR, help="Path to the directory containing instance images."
    )
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to serialize the parquet file.")
    args = parser.parse_args()

    run(args)
