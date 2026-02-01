"""
Test Interface - Core Functionality

Tests the core interface components and integration.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from interface import PhiCoordinate, LadderClosure, ResponseFunctional, HashValidator


class TestInterfaceIntegration:
    """Test full interface integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.phi = PhiCoordinate(n_levels=50, phi_max=1.0)
        self.ladder = LadderClosure(self.phi)
        self.response = ResponseFunctional(self.phi)
        self.validator = HashValidator()
    
    def test_phi_coordinate(self):
        """Test Φ coordinate system."""
        assert self.phi.n_levels == 50
        assert self.phi.phi_max == 1.0
        assert len(self.phi.phi_values) == 50
        
        # Test level indexing
        level = self.phi.get_level_index(0.5)
        assert 0 <= level < self.phi.n_levels
        
        # Test stratification
        density = np.ones(self.phi.n_levels)
        stratified = self.phi.stratify(density)
        assert stratified.shape == density.shape
    
    def test_response_functional(self):
        """Test response functional."""
        density = np.ones(self.phi.n_levels)
        
        # Test kernel response
        response = self.response.kernel_response(density)
        assert response.shape == density.shape
        
        # Test cumulative response
        cumulative = self.response.cumulative_response(density)
        assert len(cumulative) == self.phi.n_levels
        
        # Test spectral projection
        spectral = self.response.spectral_projection(density)
        assert len(spectral) == self.phi.n_levels
        
        # Test energy redistribution
        energy = self.response.energy_redistribution(response)
        assert energy.shape == response.shape
    
    def test_full_pipeline(self):
        """Test complete interface pipeline."""
        # Generate test data
        np.random.seed(123)  # Fixed seed
        density = np.random.exponential(1.0, self.phi.n_levels)
        
        # Apply ladder constraints
        constrained_density = self.ladder.apply_sigma8_containment(density)
        constrained_density = self.ladder.apply_kernel_support(constrained_density)
        
        # Compute responses
        kernel_response = self.response.kernel_response(constrained_density)
        cumulative = self.response.cumulative_response(constrained_density)
        
        # Validate outputs
        assert not np.any(np.isnan(kernel_response))
        assert not np.any(np.isnan(cumulative))
        assert len(kernel_response) == self.phi.n_levels
        assert len(cumulative) == self.phi.n_levels
        
        # Test determinism
        hash1 = self.validator.compute_hash(kernel_response)
        hash2 = self.validator.compute_hash(kernel_response)
        assert hash1 == hash2
