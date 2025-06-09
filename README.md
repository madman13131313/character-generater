# AI Game Character Generator

This project is an AI-powered tool for generating game character artwork using text-to-image diffusion models. It leverages Stable Diffusion to create visual assets that can be used in games, supporting various styles and post-processing options.

## Features

- Text-to-image generation using Stable Diffusion
- Support for different visual styles (fantasy, pixel art, anime)
- Background removal and image post-processing
- Customizable prompt modifiers
- Image resizing and cropping utilities

## Setup

1. Clone this repository

2. Set up the virtual environment:

   **For macOS/Linux:**
   ```bash
   # Make the setup script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh
   ```

   **For Windows:**
   ```bash
   setup.bat
   ```

3. Activate the virtual environment:
   
   **For macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

   **For Windows:**
   ```bash
   venv\Scripts\activate.bat
   ```

4. Create a `.env` file with your Hugging Face token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

Make sure your virtual environment is activated, then run:
```bash
python character_generator.py "your character description" --style pixel_art
```

To deactivate the virtual environment when you're done:
```bash
deactivate
```

## Project Structure

- `character_generator.py`: Main script for character generation
- `image_utils.py`: Utilities for image processing
- `requirements.txt`: Project dependencies
- `setup.sh`: Setup script for macOS/Linux
- `setup.bat`: Setup script for Windows
- `README.md`: Project documentation

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Hugging Face account and API token
