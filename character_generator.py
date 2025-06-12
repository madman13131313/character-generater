import os
import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from image_utils import apply_style_modifiers, post_process_image, save_image
import argparse
from datetime import datetime

def setup_model():
    """Initialize the Stable Diffusion model."""
    load_dotenv()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    pipe = pipe.to(device)
    
    return pipe

def generate_character(prompt, style=None, num_images=1, output_dir="output"):
    """Generate character images based on the prompt and style."""
    pipe = setup_model()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply style modifiers if specified
    if style:
        prompt = apply_style_modifiers(prompt, style)
    
    # Add game character specific modifiers
    prompt = f"{prompt}, game character, simple, centered"
    
    print(f"Generating images with prompt: {prompt}")
    
    # Generate images
    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process and save images
    for i, image in enumerate(images):
        # Save original image
        original_filename = os.path.join(output_dir, f"character_{timestamp}_{i+1}_original.png")
        save_image(image, original_filename)
        print(f"Saved original image to {original_filename}")
        
        # Process with background removal
        processed_image = post_process_image(image, remove_bg=True)
        
        # Save processed image
        processed_filename = os.path.join(output_dir, f"character_{timestamp}_{i+1}_nobg.png")
        save_image(processed_image, processed_filename)
        print(f"Saved processed image to {processed_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate game character images using Stable Diffusion")
    parser.add_argument("prompt", type=str, help="Description of the character to generate")
    parser.add_argument("--style", type=str, choices=["pixel_art", "anime", "fantasy", "cartoon"],
                      help="Visual style for the character")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    generate_character(
        args.prompt,
        style=args.style,
        num_images=args.num_images,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 