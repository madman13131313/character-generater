from PIL import Image
import numpy as np
from rembg import remove, new_session
import torch

def remove_background(image, alpha_matting=True):
    """Remove the background from an image with improved quality."""
    # Create a new session with specific model
    session = new_session("u2net")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Remove background with specific parameters
    return remove(
        image,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

def resize_image(image, target_size=(256, 256)):
    """Resize image to target dimensions while maintaining aspect ratio."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def crop_to_square(image):
    """Crop image to a square from the center."""
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))

def apply_style_modifiers(prompt, style):
    """Apply style modifiers to the prompt."""
    style_modifiers = {
        'pixel_art': 'pixel art style, 8-bit, retro game sprite',
        'anime': 'anime style, cel shaded, vibrant colors',
        'fantasy': 'fantasy art style, detailed, magical',
        'cartoon': 'cartoon style, bold outlines, vibrant colors'
    }
    
    if style in style_modifiers:
        return f"{prompt}, {style_modifiers[style]}"
    return prompt

def post_process_image(image, remove_bg=True, target_size=(256, 256)):
    """Apply all post-processing steps to the image."""
    # First resize to a larger size for better quality
    image = resize_image(image, (512, 512))
    
    if remove_bg:
        try:
            # Remove background with improved parameters
            image = remove_background(image, alpha_matting=True)
            
            # If the image is too transparent, adjust alpha channel
            if image.mode == 'RGBA':
                data = np.array(image)
                # Increase alpha values for pixels that are not fully transparent
                alpha = data[:, :, 3]
                alpha[alpha > 0] = np.clip(alpha[alpha > 0] + 50, 0, 255)
                data[:, :, 3] = alpha
                image = Image.fromarray(data)
        except Exception as e:
            print(f"Background removal failed, using original image: {e}")
            remove_bg = False
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Crop to square
    image = crop_to_square(image)
    
    # Final resize to target size
    image = resize_image(image, target_size)
    
    return image

def save_image(image, filename, format='PNG'):
    """Save image to file."""
    image.save(filename, format) 