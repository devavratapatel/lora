import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from tqdm import tqdm # Using tqdm for a nice progress bar

def auto_caption_images(image_folder, trigger_word=""):
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"Found {len(image_files)} images to caption.")

    for image_file in tqdm(image_files, desc="Generating Captions"):
        image_path = os.path.join(image_folder, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Could not open {image_file}, skipping. Error: {e}")
            continue

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=75, num_beams=5) # Increased max_length slightly
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        final_caption = f"{trigger_word}, {caption}"
        base_filename = os.path.splitext(image_file)[0]
        output_txt_path = os.path.join(image_folder, f"{base_filename}.txt")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(final_caption)

    print(f"\nAuto-captioning complete! Saved captions to individual .txt files.")


def manual_edit_captions(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"\nStarting manual caption review for {len(image_files)} images...")

    for image_file in image_files:
        base_filename = os.path.splitext(image_file)[0]
        txt_path = os.path.join(image_folder, f"{base_filename}.txt")

        if not os.path.exists(txt_path):
            print(f"Warning: No caption file found for {image_file}. Skipping.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            current_caption = f.read().strip()

        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        image.show()

        print("-" * 20)
        print(f"Editing for: {image_file}")
        print(f"Current caption: {current_caption}")
        
        new_caption = input("Enter new caption (or press Enter to keep current): ").strip()

        if new_caption:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(new_caption)
            print("Caption updated.")
        else:
            print("Caption kept as is.")
        
        if hasattr(image, 'close'):
            image.close()

    print("\nCaption editing complete!")


if __name__ == "__main__":
    image_folder = "./dataset"

    auto_caption_images(image_folder)
    
    manual_edit_captions(image_folder)