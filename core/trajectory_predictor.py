"""
IBVAP — Trajectory Prediction Engine
Predicts future positions of tracked entities using linear velocity extrapolation.
Draws predicted path lines on the video feed.
"""

import numpy as np
from typing import List, Tuple, Optional


def predict_trajectory(
    position_history: List[Tuple[int, int]],
    num_future_points: int = 15,
    lookback: int = 10,
) -> List[Tuple[int, int]]:
    """
    Predict future positions using weighted linear regression on recent positions.
    
    Args:
        position_history: List of (x, y) centroids over time
        num_future_points: Number of future points to predict
        lookback: Number of recent positions to use for velocity estimation
    
    Returns:
        List of predicted (x, y) positions
    """
    if len(position_history) < 3:
        return []
    
    # Use the most recent positions
    recent = position_history[-lookback:]
    n = len(recent)
    
    if n < 3:
        return []
    
    # Convert to numpy arrays
    positions = np.array(recent, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    
    # Weighted linear regression (more recent points have higher weight)
    weights = np.linspace(0.3, 1.0, n)
    
    # Fit x(t) and y(t) separately with weighted least squares
    try:
        # Weighted polynomial fit (degree 1 = linear)
        coeffs_x = np.polyfit(t, positions[:, 0], deg=1, w=weights)
        coeffs_y = np.polyfit(t, positions[:, 1], deg=1, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return []
    
    # Extrapolate future positions
    future_t = np.arange(n, n + num_future_points)
    future_x = np.polyval(coeffs_x, future_t)
    future_y = np.polyval(coeffs_y, future_t)
    
    predicted = [(int(x), int(y)) for x, y in zip(future_x, future_y)]
    return predicted


def compute_direction_toward_point(
    position_history: List[Tuple[int, int]],
    target: Tuple[int, int],
    lookback: int = 5,
) -> float:
    """
    Compute how strongly the entity is moving toward a target point.
    Returns a value 0.0 (moving away) to 1.0 (moving directly toward).
    
    Args:
        position_history: Recent (x, y) positions
        target: (x, y) target point (e.g., zone center)
        lookback: Number of positions to average velocity from
    
    Returns:
        Direction magnitude 0.0 to 1.0
    """
    if len(position_history) < 2:
        return 0.0
    
    recent = position_history[-lookback:]
    if len(recent) < 2:
        return 0.0
    
    # Average velocity vector
    velocities = []
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i-1][0]
        dy = recent[i][1] - recent[i-1][1]
        velocities.append((dx, dy))
    
    avg_vx = np.mean([v[0] for v in velocities])
    avg_vy = np.mean([v[1] for v in velocities])
    
    # Vector from current position to target
    current = recent[-1]
    to_target = (target[0] - current[0], target[1] - current[1])
    
    # Normalize both vectors
    vel_mag = np.sqrt(avg_vx**2 + avg_vy**2)
    target_mag = np.sqrt(to_target[0]**2 + to_target[1]**2)
    
    if vel_mag < 0.5 or target_mag < 1.0:
        return 0.0
    
    # Cosine similarity
    cos_sim = (avg_vx * to_target[0] + avg_vy * to_target[1]) / (vel_mag * target_mag)
    
    # Map from [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (cos_sim + 1) / 2))
