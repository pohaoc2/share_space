"""
CellViT Cell Segmentation Script - Simple Configuration Version
Configure all parameters directly in the script.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from PIL import Image
import torchvision.transforms as transforms
import glob


# ================== CONFIGURATION ==================
# Model settings
MODEL_PATH = "./CellViT-SAM-H-x40.pth"
GPU_ID = 0

# Input/Output
IMAGE_FOLDER = "../../../data/exp"
OUTPUT_DIR = "../../../results"

# Inference parameters
BATCH_SIZE = 8
PATCH_SIZE = 1024
OVERLAP = 64
MIXED_PRECISION = False
EXPORT_GEOJSON = True

# Cell type mapping (adjust based on your needs)
CELL_TYPES = {
    0: "background",
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial"
}
# ===================================================


def setup_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load the CellViT model checkpoint."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        print(checkpoint.keys())
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint['model']
    
    model.eval()
    model.to(device)
    print("Model loaded successfully")
    
    return model


def extract_patches(
    image_path: str, 
    patch_size: int = 1024,
    overlap: int = 64
) -> Tuple[torch.Tensor, List[Tuple], Tuple]:
    """
    Extract overlapping patches from PNG image.
    
    Returns:
        patches_tensor: Tensor of shape (N, 3, H, W)
        coordinates: List of (x_start, y_start, x_end, y_end)
        original_shape: (height, width) of original image
    """
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    print(f"Image size: {w}x{h} pixels")
    
    # Define normalization (ImageNet statistics)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    patches = []
    coordinates = []
    stride = patch_size - overlap
    
    # Extract patches with overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate boundaries
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)
            x_start = max(0, x_end - patch_size)
            y_start = max(0, y_end - patch_size)
            
            # Extract patch
            patch = img_array[y_start:y_end, x_start:x_end]
            
            # Pad if necessary
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            # Transform to tensor
            patch_pil = Image.fromarray(patch)
            patch_tensor = transform(patch_pil)
            
            patches.append(patch_tensor)
            coordinates.append((x_start, y_start, x_end, y_end))
    
    print(f"Extracted {len(patches)} patches with {overlap}px overlap")
    
    return torch.stack(patches), coordinates, (h, w)


def run_inference(
    model: torch.nn.Module,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
    mixed_precision: bool = False
) -> List:
    """Run inference on patches in batches."""
    print(f"Running inference on {patches.shape[0]} patches...")
    model.eval()
    
    all_predictions = []
    num_batches = (patches.shape[0] + batch_size - 1) // batch_size
    
    for i in range(0, patches.shape[0], batch_size):
        batch = patches[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        with torch.no_grad():
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    batch = batch.to(device)
                    pred = model(batch)
            else:
                batch = batch.to(device)
                pred = model(batch)
        
        all_predictions.append(pred)
        print(f"  Processed batch {batch_num}/{num_batches}")
    
    return all_predictions


def postprocess_predictions(
    predictions: List,
    coordinates: List[Tuple],
    original_shape: Tuple,
    overlap: int = 64
) -> Dict[str, Any]:
    """
    Postprocess and merge predictions from overlapping patches.
    
    Returns:
        Dictionary with cell instances, types, centroids, etc.
    """
    print("Postprocessing predictions...")
    
    # Placeholder for actual CellViT postprocessing
    # This should extract cell instances, types, and contours
    
    results = {
        'instances': [],
        'types': [],
        'centroids': [],
        'contours': [],
        'num_cells': 0
    }
    
    # Example: Extract information from predictions
    # Actual implementation depends on CellViT model output structure
    for idx, pred in enumerate(predictions):
        # pred might contain instance maps, type predictions, etc.
        # Extract cell information and adjust coordinates based on patch location
        x_start, y_start, x_end, y_end = coordinates[idx]
        
        # Process each prediction
        # results['centroids'].append((x + x_start, y + y_start))
        pass
    
    results['num_cells'] = len(results['centroids'])
    print(f"Detected {results['num_cells']} cells")
    
    return results


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    image_name: str,
    export_geojson: bool = True
) -> None:
    """Save detection results to disk."""
    output_path = Path(output_dir) / image_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_path}")
    
    # Save instance map
    if results['instances']:
        np.save(output_path / 'cell_instances.npy', 
                np.array(results['instances']))
    
    # Save cell types
    if results['types']:
        np.save(output_path / 'cell_types.npy', 
                np.array(results['types']))
    
    # Save centroids
    if results['centroids']:
        np.save(output_path / 'cell_centroids.npy', 
                np.array(results['centroids']))
    
    # Save summary
    summary = {
        'num_cells': results['num_cells'],
        'cell_type_counts': {},
        'image_processed': image_name
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Export GeoJSON for QuPath
    if export_geojson:
        export_geojson_file(results, output_path)
    
    print("Results saved successfully")


def export_geojson_file(results: Dict[str, Any], output_path: Path) -> None:
    """Export results to GeoJSON format for QuPath visualization."""
    features = []
    
    for i, (x, y) in enumerate(results['centroids']):
        cell_type_id = results['types'][i] if i < len(results['types']) else 0
        cell_type_name = CELL_TYPES.get(cell_type_id, 'unknown')
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(x), float(y)]
            },
            "properties": {
                "cell_id": i,
                "cell_type": cell_type_name,
                "classification": {
                    "name": cell_type_name,
                    "color": get_cell_color(cell_type_id)
                }
            }
        }
        features.append(feature)
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    geojson_path = output_path / 'cell_detection.geojson'
    with open(geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"GeoJSON exported: {geojson_path}")


def get_cell_color(cell_type_id: int) -> List[int]:
    """Return RGB color for each cell type."""
    colors = {
        0: [200, 200, 200],  # background - gray
        1: [255, 0, 0],      # neoplastic - red
        2: [0, 255, 0],      # inflammatory - green
        3: [0, 0, 255],      # connective - blue
        4: [128, 0, 128],    # dead - purple
        5: [255, 255, 0]     # epithelial - yellow
    }
    return colors.get(cell_type_id, [200, 200, 200])


def main():
    """Main execution function."""
    print("=" * 60)
    print("CellViT Cell Segmentation")
    print("=" * 60)
    
    # Setup device
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = setup_model(MODEL_PATH, device)
    
    # Extract patches from image
    patches, coordinates, original_shape = extract_patches(
        IMAGE_PATH, 
        patch_size=PATCH_SIZE, 
        overlap=OVERLAP
    )
    
    # Run inference
    predictions = run_inference(
        model, 
        patches, 
        device, 
        batch_size=BATCH_SIZE,
        mixed_precision=MIXED_PRECISION
    )
    
    # Postprocess results
    results = postprocess_predictions(
        predictions, 
        coordinates, 
        original_shape,
        overlap=OVERLAP
    )
    
    # Save results
    image_name = Path(IMAGE_PATH).stem
    save_results(
        results, 
        OUTPUT_DIR, 
        image_name,
        export_geojson=EXPORT_GEOJSON
    )
    
    print("=" * 60)
    print("Processing complete!")
    print(f"Total cells detected: {results['num_cells']}")
    print(f"Results saved to: {OUTPUT_DIR}/{image_name}/")
    print("=" * 60)


def convert_png_to_tif(image_folder: str):
    png_paths = glob.glob(os.path.join(image_folder, "*.png"))
    for png_path in png_paths:
        image = pyvips.Image.new_from_file(png_path)
        image.write_to_file(png_path.replace(".png", ".tif"), tile=True, pyramid=True, compression="jpeg")

if __name__ == '__main__':
    #main()
    image_path = '../../../data/exp/train_1_000.png'
    image = Image.open(image_path)
    print(image.size)
    asd()
    
    