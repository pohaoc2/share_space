import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cv2


def visualize_cells(image_path, json_path, output_path='cells_visualization.png'):
    """
    Visualize cell segmentation with three subplots: original, mask, and overlay.
    
    Args:
        image_path: Path to the original histopathology image
        json_path: Path to the cells.json file
        output_path: Path to save the output visualization
    """
    # Load the cell data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract information
    type_map = data['type_map']
    cells = data['cells']
    
    # Define colors for each cell type (RGB)
    colors = {
        1: (255, 107, 107),  # Neoplastic - red
        2: (78, 205, 196),   # Inflammatory - teal
        3: (69, 183, 209),   # Connective - blue
        4: (150, 206, 180),  # Dead - green
        5: (255, 234, 167),  # Epithelial - yellow
    }
    
    # Load the original image
    original_img = Image.open(image_path).convert('RGB')
    img_array = np.array(original_img)
    height, width = img_array.shape[:2]
    
    # Create mask image (filled polygons)
    mask = Image.new('RGB', (width, height), (255, 255, 255))
    draw_mask = ImageDraw.Draw(mask)
    
    for cell in cells:
        cell_type = cell['type']
        contour = np.array(cell['contour'])  # contour is in [x, y] format
        offset = np.array(cell['offset_global'])  # offset is negative, so add it
        contour_global = contour #+ offset
        
        # Convert to list of tuples for PIL (x, y format)
        polygon_coords = [(int(x), int(y)) for x, y in contour_global]
        
        # Draw filled polygon on mask
        color = colors.get(cell_type, (200, 200, 200))
        draw_mask.polygon(polygon_coords, fill=color, outline=(0, 0, 0))
    
    mask_array = np.array(mask)
    
    # Create overlay image (contours on original)
    overlay_img = img_array.copy()
    
    for cell in cells:
        cell_type = cell['type']
        contour = np.array(cell['contour'])  # contour is in [x, y] format
        offset = np.array(cell['offset_global'])  # offset is negative, so add it
        contour_global = contour# + offset
        
        # Convert to OpenCV format (needs [x, y] pairs)
        contour_cv = contour_global.astype(np.int32).reshape((-1, 1, 2))
        
        # Draw contour on overlay
        color = colors.get(cell_type, (200, 200, 200))
        cv2.drawContours(overlay_img, [contour_cv], -1, color, 2)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(img_array)
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask_array)
    axes[1].axis('off')
    
    # Plot overlay
    axes[2].imshow(overlay_img)
    axes[2].axis('off')
    
    # Remove all padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    print(f"Detected {len(cells)} cells")


if __name__ == '__main__':
    image_path = "../../../data/exp/train_1_000.png"
    json_path = "cells.json"
    output_path = 'cells_visualization.png'
    
    visualize_cells(image_path, json_path, output_path)