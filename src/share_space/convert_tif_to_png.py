"""
Convert .tif images to PNG format from the Benign folder.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def convert_tif_to_png(input_path: Path, output_path: Path, overwrite: bool = False) -> bool:
    """
    Convert a single .tif image to PNG format.
    
    Args:
        input_path: Path to the input .tif file
        output_path: Path for the output PNG file
        overwrite: If True, overwrite existing PNG files
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Check if output already exists
        if output_path.exists() and not overwrite:
            print(f"Skipping {input_path.name} - PNG already exists")
            return False
        
        # Open and convert the image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles grayscale, palette, etc.)
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Save as PNG
            img.save(output_path, 'PNG')
        
        return True
    
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def convert_benign_tif_to_png(input_folder: str, output_folder: str, overwrite: bool = False, 
                              num_workers: int = None) -> tuple[int, int]:
    """
    Convert all .tif images in the Benign folder to PNG format using parallel processing.
    
    Args:
        input_folder: Path to the input folder containing .tif files
        output_folder: Path to the output folder where PNG files will be saved
        overwrite: If True, overwrite existing PNG files
        num_workers: Number of parallel workers. If None, uses number of CPU cores
        
    Returns:
        Tuple of (successful_conversions, failed_conversions)
    """
    # Define the Benign folder path
    benign_folder = Path(input_folder)
    
    # Check if Benign folder exists
    if not benign_folder.exists():
        print(f"Error: Benign folder does not exist: {benign_folder}")
        return 0, 0
    
    # Find all .tif files in the Benign folder
    tif_files = list(benign_folder.glob('*.tif'))
    
    if not tif_files:
        print(f"No .tif files found in {benign_folder}")
        return 0, 0
    
    print(f"Found {len(tif_files)} .tif file(s) to convert")
    
    # Create output directory
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    print(f"Using {num_workers} parallel workers")
    
    successful = 0
    failed = 0
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(
                convert_tif_to_png,
                tif_path,
                output_dir / tif_path.name.replace('.tif', '.png'),
                overwrite
            ): tif_path
            for tif_path in tif_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tif_files), desc="Converting images") as pbar:
            for future in as_completed(future_to_path):
                tif_path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error processing {tif_path.name}: {e}")
                    failed += 1
                finally:
                    pbar.update(1)
    
    return successful, failed


if __name__ == '__main__':
    # Example usage: specify output folder here
    output_folder = "../../data/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/Benign_PNG"
    input_folder = "../../data/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/Benign"
    print(f"Converting .tif images from Benign folder to: {output_folder}")
    successful, failed = convert_benign_tif_to_png(input_folder, output_folder, overwrite=False)
    
    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
