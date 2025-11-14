"""
Quick GPU setup checker and PyTorch installer helper
"""

import subprocess
import sys

def check_cuda_version():
    """Check CUDA version from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        output = result.stdout
        
        # Find CUDA version
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                # Extract version number
                parts = line.split('CUDA Version:')
                if len(parts) > 1:
                    version = parts[1].strip().split()[0]
                    print(f"✓ NVIDIA GPU detected!")
                    print(f"✓ CUDA Version from drivers: {version}")
                    return version
        
        print("✓ nvidia-smi works but couldn't find CUDA version")
        return None
        
    except FileNotFoundError:
        print("✗ nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return None
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return None


def check_pytorch():
    """Check current PyTorch installation"""
    try:
        import torch
        print(f"\n✓ PyTorch is installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available in PyTorch!")
            print(f"✓ PyTorch CUDA version: {torch.version.cuda}")
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("✗ PyTorch is CPU-only (no CUDA support)")
            return False
            
    except ImportError:
        print("✗ PyTorch is not installed")
        return False


def suggest_install_command(cuda_version):
    """Suggest the right PyTorch install command"""
    
    if not cuda_version:
        print("\n📝 Install PyTorch with CUDA support:")
        print("\nVisit: https://pytorch.org/get-started/locally/")
        print("Select your preferences and copy the install command")
        return
    
    # Parse major.minor version
    major = int(cuda_version.split('.')[0])
    minor = int(cuda_version.split('.')[1])
    
    print(f"\n📝 Recommended PyTorch installation for CUDA {cuda_version}:")
    print("\n1. First uninstall current PyTorch (CPU version):")
    print("   pip uninstall torch torchvision torchaudio")
    
    print("\n2. Then install CUDA-enabled PyTorch:")
    
    if major == 11 and minor == 8:
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    elif major == 12 and minor == 1:
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    elif major == 12 and minor >= 4:
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    elif major == 11:
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    elif major == 12:
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"   # Visit https://pytorch.org for CUDA {cuda_version}")
    
    print("\n3. Verify installation:")
    print('   python -c "import torch; print(torch.cuda.is_available())"')


def main():
    print("="*60)
    print("GPU Setup Checker for PyTorch Training")
    print("="*60)
    
    # Check CUDA
    print("\n[Step 1] Checking NVIDIA GPU and CUDA drivers...")
    cuda_version = check_cuda_version()
    
    # Check PyTorch
    print("\n[Step 2] Checking PyTorch installation...")
    has_cuda = check_pytorch()
    
    # Provide recommendations
    print("\n" + "="*60)
    if has_cuda:
        print("✓ All set! Your GPU is ready for training.")
        print("\nYou can now run:")
        print("  python train.py")
    else:
        print("⚠ Action needed: Install CUDA-enabled PyTorch")
        suggest_install_command(cuda_version)
        
    print("="*60)
    
    # Show config optimization
    print("\n💡 Configuration optimized for 4GB GPU:")
    print("   - Sequence length: 128")
    print("   - Batch size: 64")
    print("   - Transformer layers: 4")
    print("   - Expected GPU usage: ~3.5GB")
    print("\n   Training will be ~25-80x faster on GPU!")
    print("="*60)


if __name__ == "__main__":
    main()
