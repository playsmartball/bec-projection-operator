"""
CPU Execution Script - Deterministic FMI Interface

Runs the FMI interface on CPU with deterministic output validation.
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('..')

from interface import PhiCoordinate, LadderClosure, ResponseFunctional, HashValidator


def main():
    """Main CPU execution with deterministic output."""
    print("Running FMI interface on CPU...")
    
    # Initialize components
    phi = PhiCoordinate(n_levels=1000, phi_max=1.0)
    ladder = LadderClosure(phi)
    response = ResponseFunctional(phi)
    validator = HashValidator()
    
    # Generate deterministic test data
    np.random.seed(42)  # Fixed seed for determinism
    density = np.random.exponential(1.0, phi.n_levels)
    
    # Apply ladder constraints
    density = ladder.apply_sigma8_containment(density)
    density = ladder.apply_kernel_support(density)
    
    # Compute response
    kernel_response = response.kernel_response(density)
    cumulative = response.cumulative_response(density)
    spectral = response.spectral_projection(density)
    energy = response.energy_redistribution(kernel_response)
    
    # Create output dictionary
    output = {
        "phi_values": phi.phi_values.tolist(),
        "density": density.tolist(),
        "kernel_response": kernel_response.tolist(),
        "cumulative_response": cumulative.tolist(),
        "spectral_magnitude": np.abs(spectral).tolist(),
        "energy_redistribution": energy.tolist(),
        "metadata": {
            "execution": "cpu",
            "n_levels": phi.n_levels,
            "seed": 42,
            "fmi_version": "0.1.0"
        }
    }
    
    # Save output
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "cpu_output.json"
    json.dump(output, open(output_file, 'w'), indent=2)
    
    # Save reference trace
    trace_data = np.array(output["kernel_response"])
    validator.save_reference(trace_data, "phi_trace")
    
    # Compute and display hash
    output_hash = validator.compute_dict_hash(output)
    print(f"CPU execution complete. Output hash: {output_hash}")
    print(f"Output saved to: {output_file}")
    print(f"Reference trace saved to: data/reference/phi_trace.*")
    
    return output_hash


if __name__ == "__main__":
    main()
