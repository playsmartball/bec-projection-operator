"""
Phi Coordinate - Depth Stratification Interface

Implements the Φ depth coordinate as a deterministic, ordered parameter space.
"""

import numpy as np
from typing import Tuple, List


class PhiCoordinate:
    """
    Depth coordinate Φ for stratified systems.
    
    Φ is an ordering parameter, not time. It provides a deterministic
    stratification of the system state space.
    """
    
    def __init__(self, n_levels: int = 1000, phi_max: float = 1.0):
        """
        Initialize Φ coordinate system.
        
        Args:
            n_levels: Number of discrete Φ levels
            phi_max: Maximum Φ value (normalized)
        """
        self.n_levels = n_levels
        self.phi_max = phi_max
        self.phi_values = np.linspace(0, phi_max, n_levels)
        self.dphi = phi_max / (n_levels - 1)
        
    def get_level_index(self, phi: float) -> int:
        """Convert Φ value to discrete level index."""
        return int(np.floor(phi / self.dphi))
    
    def get_phi_range(self, level: int) -> Tuple[float, float]:
        """Get Φ range for a given level."""
        phi_start = level * self.dphi
        phi_end = min((level + 1) * self.dphi, self.phi_max)
        return phi_start, phi_end
    
    def stratify(self, density: np.ndarray) -> np.ndarray:
        """
        Apply Φ stratification to density array.
        
        Args:
            density: Input density distribution
            
        Returns:
            Stratified density over Φ levels
        """
        if len(density) != self.n_levels:
            raise ValueError(f"Density length {len(density)} != n_levels {self.n_levels}")
        
        return density * self.phi_values
    
    def cumulative_integral(self, density: np.ndarray) -> np.ndarray:
        """
        Compute cumulative integral over Φ.
        
        Args:
            density: Density distribution over Φ
            
        Returns:
            Cumulative integral at each Φ level
        """
        return np.cumsum(density) * self.dphi
