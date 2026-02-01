"""
Test Ladder Closure - Constraint Validation

Tests the ladder closure system and constraint enforcement.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from interface import PhiCoordinate, LadderClosure


class TestLadderClosure:
    """Test ladder closure functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.phi = PhiCoordinate(n_levels=100, phi_max=1.0)
        self.ladder = LadderClosure(self.phi)
    
    def test_initialization(self):
        """Test ladder initialization."""
        assert self.ladder.rungs_completed == 17
        assert self.ladder.validate_closure() == True
        assert self.ladder.constraints['sigma8_containment'] == True
    
    def test_sigma8_containment(self):
        """Test σ₈ containment constraint."""
        # Create high-variance density
        density = np.random.normal(0, 10, self.phi.n_levels)
        
        # Apply constraint
        contained = self.ladder.apply_sigma8_containment(density)
        
        # Check variance is normalized
        assert np.var(contained) <= 1.0 + 1e-10
    
    def test_kernel_support(self):
        """Test kernel support constraint."""
        density = np.ones(self.phi.n_levels)
        
        # Apply kernel support
        supported = self.ladder.apply_kernel_support(density)
        
        # Check windowing effect
        assert np.max(supported) <= 1.0
        assert np.all(np.diff(supported) <= 1e-10)  # Smooth
    
    def test_orthogonality(self):
        """Test k-orthogonality enforcement."""
        # Create non-orthogonal modes
        modes = np.array([
            np.ones(self.phi.n_levels),
            np.ones(self.phi.n_levels) + 0.1 * np.random.random(self.phi.n_levels)
        ])
        
        # Apply orthogonality
        orthogonal = self.ladder.enforce_orthogonality(modes)
        
        # Check orthogonality
        dot_product = np.dot(orthogonal[0], orthogonal[1])
        assert abs(dot_product) < 1e-10
