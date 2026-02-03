"""
Φ-Integrity Projection Module (Fork A - Locked)

Fixed-dimensional Φ projection with locked parameters.
No tuning. No magic. Deterministic by construction.
"""

import hashlib
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


# LOCKED PARAMETERS - DO NOT CHANGE
PHI_RESOLUTION = 5000  # Fixed dimensional output
PHI_RANGE = (0.0, 10.0)  # Fixed Φ value range
PHI_KERNEL_SIGMA = (PHI_RANGE[1] - PHI_RANGE[0]) / 10.0  # Fixed kernel width


@dataclass
class ProjectionMetrics:
    """Metrics computed during Φ projection."""
    dimensional_saturation: float
    entropy_loss: float
    variance_concentration: float
    sparsity_ratio: float
    information_density: float
    projection_efficiency: float


class PhiProjection:
    """
    Fixed-dimensional Φ projection operator.
    
    LOCKED SPECIFICATION:
    - Fixed resolution: 5000
    - Fixed range: (0.0, 10.0)
    - Fixed kernel: Gaussian with sigma = 1.0
    - No parameter tuning allowed
    """
    
    def __init__(self):
        """Initialize with locked parameters."""
        self.resolution = PHI_RESOLUTION
        self.phi_range = PHI_RANGE
        self.sigma = PHI_KERNEL_SIGMA
        
        # Pre-compute fixed kernel (deterministic)
        self._setup_fixed_kernel()
    
    def _setup_fixed_kernel(self):
        """Setup fixed Gaussian kernel - no tuning allowed."""
        # Create Φ values
        self.phi_values = np.linspace(self.phi_range[0], self.phi_range[1], self.resolution)
        
        # Create fixed Gaussian kernel
        diff = self.phi_values[:, np.newaxis] - self.phi_values
        self.kernel = np.exp(-diff**2 / (2 * self.sigma**2))
        
        # Pre-compute orthogonalization matrix (fixed)
        self.ortho_matrix = self._compute_fixed_orthogonalization()
    
    def _compute_fixed_orthogonalization(self) -> np.ndarray:
        """Compute fixed orthogonalization matrix."""
        # Simple Gram-Schmidt orthogonalization
        Q = np.eye(self.resolution)
        for i in range(self.resolution):
            for j in range(i):
                Q[i] = Q[i] - np.dot(Q[i], Q[j]) * Q[j]
            norm = np.linalg.norm(Q[i])
            if norm > 1e-10:
                Q[i] = Q[i] / norm
        return Q
    
    def _raw_byte_ingestion(self, data: str) -> np.ndarray:
        """
        Raw byte ingestion - no preprocessing magic.
        
        Args:
            data: Input string
            
        Returns:
            Raw byte array
        """
        return np.frombuffer(data.encode('utf-8'), dtype=np.uint8)
    
    def _apply_kernel_support(self, x: np.ndarray) -> np.ndarray:
        """
        Apply kernel support - pad/truncate to fixed resolution.
        
        Args:
            x: Input array
            
        Returns:
            Fixed-size array
        """
        if len(x) > self.resolution:
            # Truncate and apply window
            x = x[:self.resolution]
            window = np.exp(-((np.arange(len(x)) - len(x)/2) / (len(x)/4)) ** 2)
            return x * window
        elif len(x) < self.resolution:
            # Pad with zeros
            return np.pad(x, (0, self.resolution - len(x)), mode='constant')
        else:
            # Apply window
            window = np.exp(-((np.arange(len(x)) - len(x)/2) / (len(x)/4)) ** 2)
            return x * window
    
    def _project_to_phi_space(self, x: np.ndarray) -> np.ndarray:
        """
        Project to Φ-space using fixed kernel.
        
        Args:
            x: Input array
            
        Returns:
            Φ-space projection
        """
        # Apply kernel support
        x_supported = self._apply_kernel_support(x)
        
        # Project to Φ-space
        phi_projection = np.dot(self.kernel, x_supported)
        
        # Apply orthogonalization
        phi_orthogonal = np.dot(self.ortho_matrix, phi_projection)
        
        return phi_orthogonal
    
    def _calculate_shannon_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        # Create histogram
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero entries
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _compute_metrics(self, 
                        input_data: np.ndarray,
                        phi_projection: np.ndarray) -> ProjectionMetrics:
        """
        Compute projection metrics.
        
        Args:
            input_data: Original input
            phi_projection: Φ-space projection
            
        Returns:
            Projection metrics
        """
        # Calculate entropies
        input_entropy = self._calculate_shannon_entropy(input_data)
        phi_entropy = self._calculate_shannon_entropy(phi_projection)
        
        # Dimensional saturation
        phi_std = np.std(phi_projection)
        dimensional_saturation = min(1.0, phi_std / 10.0)
        
        # Entropy loss
        entropy_loss = max(0.0, input_entropy - phi_entropy)
        
        # Variance concentration
        input_variance = np.var(input_data)
        phi_variance = np.var(phi_projection)
        variance_concentration = min(1.0, phi_variance / (input_variance + 1e-10))
        
        # Sparsity ratio
        sparsity_threshold = 1e-6
        sparsity_ratio = np.sum(np.abs(phi_projection) < sparsity_threshold) / len(phi_projection)
        
        # Information density
        information_density = phi_entropy / len(phi_projection)
        
        # Projection efficiency
        projection_efficiency = 1.0 - (entropy_loss / (input_entropy + 1e-10))
        
        return ProjectionMetrics(
            dimensional_saturation=dimensional_saturation,
            entropy_loss=entropy_loss,
            variance_concentration=variance_concentration,
            sparsity_ratio=sparsity_ratio,
            information_density=information_density,
            projection_efficiency=projection_efficiency
        )
    
    def project(self, data: str) -> Tuple[np.ndarray, ProjectionMetrics, str]:
        """
        Project data to Φ-space with locked parameters.
        
        Args:
            data: Input string
            
        Returns:
            Tuple of (phi_projection, metrics, phi_hash)
        """
        # Raw byte ingestion
        raw_bytes = self._raw_byte_ingestion(data)
        
        # Convert to float for projection
        raw_float = raw_bytes.astype(np.float64)
        
        # Project to Φ-space
        phi_projection = self._project_to_phi_space(raw_float)
        
        # Compute metrics
        metrics = self._compute_metrics(raw_float, phi_projection)
        
        # Compute hash for reproducibility
        phi_hash = hashlib.sha256(phi_projection.tobytes()).hexdigest()[:16]
        
        return phi_projection, metrics, phi_hash
    
    def get_locked_parameters(self) -> Dict[str, Any]:
        """Get locked parameters (for verification)."""
        return {
            "resolution": self.resolution,
            "phi_range": self.phi_range,
            "sigma": self.sigma,
            "kernel_shape": self.kernel.shape,
            "kernel_determinant": np.linalg.det(self.kernel[:10, :10])  # Sample for verification
        }


# Global instance (locked, singleton)
_PHI_PROJECTION = PhiProjection()


def project_to_phi(data: str) -> Tuple[np.ndarray, ProjectionMetrics, str]:
    """
    Global function for Φ projection.
    
    Args:
        data: Input string
        
    Returns:
        Tuple of (phi_projection, metrics, phi_hash)
    """
    return _PHI_PROJECTION.project(data)


def get_locked_spec() -> Dict[str, Any]:
    """Get locked specification (for verification)."""
    return _PHI_PROJECTION.get_locked_parameters()
