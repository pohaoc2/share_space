"""
Histopathology Image Segmentation Pipeline
Supports: CellPose, Mesmer, StarDist
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from skimage import io


class SegmentationModel(ABC):
    """Base class for segmentation models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
    
    @abstractmethod
    def load_model(self, **kwargs):
        """Load the model with specific parameters"""
        pass
    
    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Segment the image
        
        Args:
            image: Input image (H, W, C) for RGB or (H, W) for grayscale
            
        Returns:
            Instance segmentation mask (H, W) with integer labels
        """
        pass
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience method for segmentation"""
        return self.segment(image, **kwargs)


class CellPoseSegmenter(SegmentationModel):
    """CellPose segmentation wrapper"""
    
    def __init__(self):
        super().__init__("CellPose")
        
    def load_model(self, model_type: str = 'cyto2', gpu: bool = True, **kwargs):
        """
        Load CellPose model
        
        Args:
            model_type: 'cyto', 'cyto2', 'nuclei', or path to custom model
            gpu: Use GPU acceleration
        """
        from cellpose import models
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)
        
    def segment(self, image: np.ndarray, diameter: Optional[float] = None, 
                channels: list = [0, 0], **kwargs) -> np.ndarray:
        """
        Segment with CellPose
        
        Args:
            image: Input image
            diameter: Expected cell diameter (None for auto)
            channels: [cytoplasm_channel, nucleus_channel], use [0,0] for grayscale
        """
        if self.model is None:
            self.load_model()
            
        masks, flows, styles, diams = self.model.eval(
            image, 
            diameter=diameter, 
            channels=channels,
            **kwargs
        )
        return masks


class MesmerSegmenter(SegmentationModel):
    """Mesmer segmentation wrapper"""
    
    def __init__(self):
        super().__init__("Mesmer")
        
    def load_model(self, **kwargs):
        """Load Mesmer model from DeepCell"""
        from deepcell.applications import Mesmer
        self.model = Mesmer()
        
    def segment(self, image: np.ndarray, 
                compartment: str = 'whole-cell',
                image_mpp: float = 0.5,
                **kwargs) -> np.ndarray:
        """
        Segment with Mesmer
        
        Args:
            image: Input image (H, W, 2) - channel 0: nuclear, channel 1: membrane
            compartment: 'whole-cell', 'nuclear', or 'both'
            image_mpp: Microns per pixel resolution
        """
        if self.model is None:
            self.load_model()
        
        # Mesmer expects (batch, H, W, channels)
        if image.ndim == 3 and image.shape[-1] == 2:
            img_input = np.expand_dims(image, axis=0)
        else:
            raise ValueError("Mesmer requires 2-channel image (nuclear, membrane)")
            
        masks = self.model.predict(
            img_input,
            image_mpp=image_mpp,
            compartment=compartment,
            **kwargs
        )
        return masks[0, ..., 0]  # Return first batch, first channel


class StarDistSegmenter(SegmentationModel):
    """StarDist segmentation wrapper"""
    
    def __init__(self):
        super().__init__("StarDist")
        
    def load_model(self, model_name: str = '2D_versatile_fluo', **kwargs):
        """
        Load StarDist model
        
        Args:
            model_name: Pre-trained model name or path to custom model
                       '2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'
        """
        from stardist.models import StarDist2D
        self.model = StarDist2D.from_pretrained(model_name)
        
    def segment(self, image: np.ndarray, 
                prob_thresh: float = 0.5,
                nms_thresh: float = 0.4,
                **kwargs) -> np.ndarray:
        """
        Segment with StarDist
        
        Args:
            image: Input image (H, W) grayscale or (H, W, C) RGB
            prob_thresh: Probability threshold for detection
            nms_thresh: Non-maximum suppression threshold
        """
        if self.model is None:
            self.load_model()
        
        # Convert to grayscale if RGB
        if image.ndim == 3:
            image = np.mean(image, axis=-1)
            
        masks, details = self.model.predict_instances(
            image,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            **kwargs
        )
        return masks


class SegmentationPipeline:
    """Main pipeline for running multiple segmentation models"""
    
    def __init__(self):
        self.models: Dict[str, SegmentationModel] = {}
        
    def add_model(self, model: SegmentationModel):
        """Add a segmentation model to the pipeline"""
        self.models[model.model_name] = model
        
    def segment_image(self, image: np.ndarray, 
                     model_name: str,
                     **kwargs) -> np.ndarray:
        """
        Segment image with specified model
        
        Args:
            image: Input image
            model_name: Name of the model to use
            **kwargs: Model-specific parameters
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].segment(image, **kwargs)
    
    def segment_all(self, image: np.ndarray, 
                   model_configs: Optional[Dict[str, Dict]] = None) -> Dict[str, np.ndarray]:
        """
        Segment image with all available models
        
        Args:
            image: Input image
            model_configs: Dict mapping model names to their kwargs
        """
        if model_configs is None:
            model_configs = {name: {} for name in self.models.keys()}
            
        results = {}
        for model_name in self.models.keys():
            kwargs = model_configs.get(model_name, {})
            results[model_name] = self.segment_image(image, model_name, **kwargs)
            
        return results


# Example usage
if __name__ == "__main__":
    # Load image
    image_path = "../../../data/exp/train_1_000.png"
    image = io.imread(image_path)
    
    # Initialize pipeline
    pipeline = SegmentationPipeline()
    
    # Add CellPose
    cellpose = CellPoseSegmenter()
    cellpose.load_model(model_type='nuclei', gpu=True)
    pipeline.add_model(cellpose)
    
    # Add StarDist
    stardist = StarDistSegmenter()
    stardist.load_model(model_name='2D_versatile_he')  # H&E model
    pipeline.add_model(stardist)
    
    # Add Mesmer (if you have 2-channel input)
    # mesmer = MesmerSegmenter()
    # mesmer.load_model()
    # pipeline.add_model(mesmer)
    
    # Segment with single model
    masks_cellpose = pipeline.segment_image(
        image, 
        "CellPose",
        diameter=30,
        channels=[0, 0]
    )
    
    # Segment with all models
    all_masks = pipeline.segment_all(
        image,
        model_configs={
            "CellPose": {"diameter": 30, "channels": [0, 0]},
            "StarDist": {"prob_thresh": 0.5}
        }
    )
    
    print(f"CellPose found {masks_cellpose.max()} cells")
    print(f"StarDist found {all_masks['StarDist'].max()} cells")