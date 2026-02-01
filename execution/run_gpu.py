"""
GPU Execution Script - Deterministic FMI Interface

Runs the FMI interface on GPU with deterministic output validation.
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('..')

from interface import PhiCoordinate, LadderClosure, ResponseFunctional, HashValidator


def main():
    """Main GPU execution with deterministic output."""
    print("Running FMI interface on GPU...")
    
    # Check GPU availability
    try:
        import cupy as cp
        gpu_available = True
        print("GPU detected: Using CuPy")
    except ImportError:
        print("GPU not available: Falling back to NumPy")
        import numpy as cp
        gpu_available = False
    
    # Initialize components
    phi = PhiCoordinate(n_levels=1000, phi_max=1.0)
    ladder = LadderClosure(phi)
    response = ResponseFunctional(phi)
    validator = HashValidator()
    
    # Generate deterministic test data (same seed as CPU)
    np.random.seed(42)  # Fixed seed for determinism
    density = np.random.exponential(1.0, phi.n_levels)
    
    # Transfer to GPU if available
    if gpu_available:
        density = cp.asarray(density)
        phi_values = cp.asarray(phi.phi_values)
    else:
        phi_values = phi.phi_values
    
    # Apply ladder constraints
    density = ladder.apply_sigma8_containment(density)
    density = ladder.apply_kernel_support(density)
    
    # Compute response
    kernel_response = response.kernel_response(density)
    cumulative = response.cumulative_response(density)
    spectral = response.spectral_projection(density)
    energy = response.energy_redistribution(kernel_response)
    
    # Transfer back to CPU for saving
    if gpu_available:
        kernel_response = cp.asnumpy(kernel_response)
        cumulative = cp.asnumpy(cumulative)
        spectral = cp.asnumpy(spectral)
        energy = cp.asnumpy(energy)
        density = cp.asnumpy(density)
    
    # Create output dictionary
    output = {
        "phi_values": phi.phi_values.tolist(),
        "density": density.tolist(),
        "kernel_response": kernel_response.tolist(),
        "cumulative_response": cumulative.tolist(),
        "spectral_magnitude": np.abs(spectral).tolist(),
        "energy_redistribution": energy.tolist(),
        "metadata": {
            "execution": "gpu" if gpu_available else "cpu_fallback",
            "n_levels": phi.n_levels,
            "seed": 42,
            "fmi_version": "0.1.0"
        }
    }
    
    # Save output
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "gpu_output.json"
    json.dump(output, open(output_file, 'w'), indent=2)
    
    # Compute and display hash
    output_hash = validator.compute_dict_hash(output)
    print(f"GPU execution complete. Output hash: {output_hash}")
    print(f"Output saved to: {output_file}")
    
    return output_hash


if __name__ == "__main__":
    main()
