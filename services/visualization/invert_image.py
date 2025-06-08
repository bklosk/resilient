\
# filepath: /workspaces/photogrammetry/services/visualization/invert_image.py
\"\"\"Invert the colors of an image.\"\"\"

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
import numpy as np


def invert_image_colors(image_path: str, output_dir: Optional[str] = None) -> str:
    \"\"\"
    Invert the colors of an image, preserving the alpha channel.

    Args:
        image_path: Path to the input PNG image file.
        output_dir: Output directory for the inverted PNG file. 
                    Defaults to the same directory as the input image, 
                    with "_inverted" appended to the original filename.

    Returns:
        str: Path to the generated inverted PNG file.
    \"\"\"
    src_path = Path(image_path)
    
    if output_dir is None:
        output_dir_path = src_path.parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    out_path = output_dir_path / f"{src_path.stem}_inverted{src_path.suffix}"

    try:
        img = Image.open(src_path)
        
        if img.mode == 'RGBA':
            # Separate RGB and Alpha channels
            rgb = img.convert('RGB')
            alpha = img.split()[-1]
            
            # Invert RGB channels
            inverted_rgb = ImageOps.invert(rgb)
            
            # Merge inverted RGB with original Alpha
            inverted_img = Image.merge('RGBA', (*inverted_rgb.split(), alpha))
            
        elif img.mode == 'RGB':
            inverted_img = ImageOps.invert(img)
        else:
            # For other modes, convert to RGB first, invert, then convert back if necessary
            # This example will just convert to RGB and invert.
            # For more complex modes, specific handling might be needed.
            rgb_img = img.convert('RGB')
            inverted_img = ImageOps.invert(rgb_img)
            # If original mode was L (grayscale), convert back
            if img.mode == 'L':
                inverted_img = inverted_img.convert('L')

        # Save the inverted image
        inverted_img.save(out_path, optimize=False, compress_level=0)
        
        print(f"Generated inverted image: {out_path}")
        print(f"Original image mode: {img.mode}")
        print(f"Inverted image mode: {inverted_img.mode}")
        print(f"Inverted image resolution: {inverted_img.size[0]}x{inverted_img.size[1]} pixels")
        print(f"Inverted image file size: {Path(out_path).stat().st_size} bytes")

        return str(out_path)

    except FileNotFoundError:
        print(f"Error: Input image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"An error occurred during image inversion: {e}")
        raise

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # Create a dummy RGBA image for testing
    dummy_array = np.zeros((100, 100, 4), dtype=np.uint8)
    dummy_array[25:75, 25:75, 0] = 255  # Red channel
    dummy_array[25:75, 25:75, 1] = 128  # Green channel
    dummy_array[25:75, 25:75, 2] = 0    # Blue channel
    dummy_array[25:75, 25:75, 3] = 150  # Alpha channel (semi-transparent)
    dummy_array[0:25, 0:25, :] = [50,100,150,255] # another color with full alpha
    
    img = Image.fromarray(dummy_array, 'RGBA')
    test_image_path = Path("./test_image_rgba.png")
    img.save(test_image_path)

    print(f"Created dummy test image: {test_image_path}")

    # Test RGBA inversion
    try:
        inverted_path_rgba = invert_image_colors(str(test_image_path))
        print(f"RGBA Inversion test successful. Inverted image: {inverted_path_rgba}")
        # Clean up dummy inverted image
        # Path(inverted_path_rgba).unlink(missing_ok=True)
    except Exception as e:
        print(f"RGBA Inversion test failed: {e}")

    # Create a dummy RGB image
    dummy_rgb_array = np.array(img.convert('RGB'))
    img_rgb = Image.fromarray(dummy_rgb_array, 'RGB')
    test_image_rgb_path = Path("./test_image_rgb.png")
    img_rgb.save(test_image_rgb_path)
    print(f"Created dummy RGB test image: {test_image_rgb_path}")

    # Test RGB inversion
    try:
        inverted_path_rgb = invert_image_colors(str(test_image_rgb_path))
        print(f"RGB Inversion test successful. Inverted image: {inverted_path_rgb}")
        # Clean up dummy inverted image
        # Path(inverted_path_rgb).unlink(missing_ok=True)
    except Exception as e:
        print(f"RGB Inversion test failed: {e}")
    
    # Clean up dummy images
    # test_image_path.unlink(missing_ok=True)
    # test_image_rgb_path.unlink(missing_ok=True)

    print("To view images, open test_image_rgba.png, test_image_rgb.png and their inverted versions.")
