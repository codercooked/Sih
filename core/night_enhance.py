"""
IBVAP — Night-Time / Low-Light Enhancement Module
Uses OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization).

LIMITATION: This is software enhancement, NOT thermal/IR imaging.
Works on grayscale channel to improve contrast in dark frames.
"""

import cv2
import numpy as np
from typing import Tuple


class NightEnhancer:
    """
    Detects low-light conditions and applies CLAHE enhancement.
    
    Honest limitation: This is software-only contrast enhancement.
    It is NOT equivalent to thermal or infrared imaging.
    """

    def __init__(
        self,
        brightness_threshold: int = 80,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        """
        Args:
            brightness_threshold: Mean pixel value below which CLAHE is applied
            clip_limit: CLAHE contrast limiting parameter
            tile_grid_size: CLAHE tile grid size
        """
        self.brightness_threshold = brightness_threshold
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def check_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate mean brightness of a frame.

        Args:
            frame: BGR image

        Returns:
            Mean brightness value (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_low_light(self, frame: np.ndarray) -> bool:
        """Check if frame is below brightness threshold."""
        return self.check_brightness(frame) < self.brightness_threshold

    def enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Conditionally enhance a frame if low-light is detected.

        Uses LAB color space to enhance luminance while preserving color.

        Args:
            frame: BGR image

        Returns:
            Tuple of (enhanced_frame, is_night_mode)
        """
        brightness = self.check_brightness(frame)

        if brightness >= self.brightness_threshold:
            return frame, False

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L-channel (luminance)
        enhanced_l = self.clahe.apply(l_channel)

        # Merge back
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Check if enhancement made the frame too noisy (optional quality check)
        enhanced_brightness = self.check_brightness(enhanced_frame)
        if enhanced_brightness < 30:
            # Enhancement didn't help much, return original
            return frame, True

        return enhanced_frame, True
