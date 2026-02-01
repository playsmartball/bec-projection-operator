"""
CLI Interface for Φ-Compressor

Command-line interface for compression and decompression operations.
"""

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.append('..')

from .compressor import PhiCompressor
from .decompressor import PhiDecompressor


def compress_command(args):
    """Handle compression command."""
    print(f"Compressing: {args.input}")
    print(f"Φ range: {args.phi}")
    print(f"Resolution: {args.resolution}")
    print(f"Backend: {args.backend}")
    
    # Parse Φ range
    phi_min, phi_max = map(float, args.phi)
    
    # Initialize compressor
    compressor = PhiCompressor(
        phi_range=(phi_min, phi_max),
        resolution=args.resolution,
        backend=args.backend
    )
    
    # Measure performance
    start_time = time.time()
    
    try:
        metadata = compressor.compress_file(args.input, args.out)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Display results
        print(f"\n✅ Compression completed in {duration:.2f} seconds")
        print(f"Compression ratio: {metadata['compression_metrics']['compression_ratio']:.2f}x")
        print(f"Size reduction: {metadata['compression_metrics']['size_reduction_percent']:.1f}%")
        print(f"Original size: {metadata['compression_metrics']['original_size_bytes']:,} bytes")
        print(f"Compressed size: {metadata['compression_metrics']['compressed_size_bytes']:,} bytes")
        
        # Save performance metadata
        perf_metadata = {
            "performance": {
                "compression_time_seconds": duration,
                "backend_used": metadata["projection_metadata"]["backend"]
            },
            **metadata
        }
        
        perf_path = Path(args.out) / f"{Path(args.input).stem}_performance.json"
        with open(perf_path, 'w') as f:
            json.dump(perf_metadata, f, indent=2)
        
        print(f"Results saved to: {args.out}")
        
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        sys.exit(1)


def decompress_command(args):
    """Handle decompression command."""
    print(f"Decompressing: {args.input}")
    
    # Initialize decompressor
    decompressor = PhiDecompressor()
    
    # Measure performance
    start_time = time.time()
    
    try:
        # Find manifest file
        input_path = Path(args.input)
        manifest_path = input_path.parent / f"{input_path.stem.replace('_compressed', '')}_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        metadata = decompressor.decompress_file(args.input, manifest_path, args.out)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Display results
        print(f"\n✅ Decompression completed in {duration:.2f} seconds")
        
        if metadata["reconstruction_metrics"]["reconstruction_error"] is not None:
            error = metadata["reconstruction_metrics"]["reconstruction_error"]
            print(f"Reconstruction error: {error:.6e}")
        else:
            print("Reconstruction error: Original file not available for comparison")
        
        print(f"Reconstructed shape: {metadata['reconstruction_metrics']['reconstructed_shape']}")
        print(f"Reconstructed size: {metadata['reconstruction_metrics']['reconstructed_size_bytes']:,} bytes")
        print(f"Output saved to: {args.out}")
        
    except Exception as e:
        print(f"❌ Decompression failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Φ-Compressor: Deterministic numeric data compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a file
  python -m bec_phi_compress.cli compress \\
    --input data.npy \\
    --phi 0.0 10.0 \\
    --resolution 5000 \\
    --backend gpu \\
    --out output/compressed_run
  
  # Decompress a file
  python -m bec_phi_compress.cli decompress \\
    --input output/compressed_run/data_compressed.npy \\
    --out output/reconstructed.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compression command
    compress_parser = subparsers.add_parser('compress', help='Compress a numeric file')
    compress_parser.add_argument('--input', required=True, help='Input file path')
    compress_parser.add_argument('--phi', nargs=2, required=True, help='Φ range (min max)')
    compress_parser.add_argument('--resolution', type=int, default=5000, help='Φ resolution')
    compress_parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'], default='auto', help='Computational backend')
    compress_parser.add_argument('--out', required=True, help='Output directory')
    compress_parser.set_defaults(func=compress_command)
    
    # Decompression command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a file')
    decompress_parser.add_argument('--input', required=True, help='Compressed file path')
    decompress_parser.add_argument('--out', required=True, help='Output file path')
    decompress_parser.set_defaults(func=decompress_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
