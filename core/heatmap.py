"""
IBVAP — Spatial Heatmap Accumulator
Tracks entity movement over time and generates a 2D heatmap overlay.
"""

import cv2
import numpy as np

class HeatmapAccumulator:
    def __init__(self, width: int, height: int, decay_rate: float = 0.995):
        """
        Args:
            width: Frame width
            height: Frame height
            decay_rate: How fast the heatmap fades over time (1.0 = never fade)
        """
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        # Create a float32 matrix to accumulate heat
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def add_point(self, x: int, y: int, radius: int = 40, intensity: float = 0.1):
        """Add heat at a specific point."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
            
        # Draw a circle on a temporary mask to add smooth heat
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        cv2.circle(mask, (x, y), radius, intensity, -1)
        
        # Apply Gaussian blur to the mask to make the heat look natural
        mask = cv2.GaussianBlur(mask, (radius | 1, radius | 1), 0) # radius | 1 ensures odd size
        
        self.heatmap += mask

    def update(self):
        """Apply decay every frame."""
        self.heatmap *= self.decay_rate

    def get_overlay(self) -> np.ndarray:
        """Get the heatmap colored as a BGR image."""
        # Normalize heatmap to 0-255
        norm_heat = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap (JET goes from blue -> green -> yellow -> red)
        color_heatmap = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        
        # Make the cold areas (blue) black so they can be transparent
        # In JET, cold areas are usually (255, 0, 0)
        mask = norm_heat > 5
        result = np.zeros_like(color_heatmap)
        result[mask] = color_heatmap[mask]
        
        return result
        
    def blend(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend the heatmap over the video frame."""
        overlay = self.get_overlay()
        
        # Only blend where heatmap has data
        gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        mask = gray_overlay > 0
        
        # Blend the entire frame with overlay using OpenCV (which handles types safely)
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Restore the original pixels where there is no heat (mask is 0)
        # We invert the mask using ~ to select the "cold" areas
        blended[~mask] = frame[~mask]
        
        return blended
