"""Quick test loading a single model to verify functionality."""

import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

# Add project to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO, SAC  # noqa: E402


def test_single_model(model_zip: str, algo: str = "ppo"):
    """Test loading a single model."""
    zip_path = ROOT_DIR / model_zip
    
    print(f"\n{'='*60}")
    print(f"Testing Model Load: {model_zip}")
    print(f"Algorithm: {algo.upper()}")
    print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    model_class = PPO if algo == "ppo" else SAC
    
    # Create a temporary directory with proper permissions
    temp_dir = Path(tempfile.mkdtemp())
    try:
        extract_dir = temp_dir / "model"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting to: {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        
        # List files
        files = list(extract_dir.glob("*"))
        print(f"Extracted files: {len(files)}")
        for f in files:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name}: {size_kb:.1f} KB")
        
        # Try loading with allow_pickle=True
        print("\nAttempting to load model...")
        os.chdir(extract_dir)  # Change to extraction directory
        model = model_class.load(str(extract_dir), device="cpu", allow_pickle=True)
        
        print("\n✓ SUCCESS!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Total timesteps: {model.num_timesteps:,}")
        
        # Try a quick predict
        if hasattr(model, 'get_env') and model.get_env() is not None:
            obs = model.get_env().reset()
            action, _ = model.predict(obs, deterministic=True)
            print(f"  Quick predict: OK (action shape: {action.shape if hasattr(action, 'shape') else 'scalar'})")
        
        return True
        
    except Exception as e:
        print("\n✗ FAILED!")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        os.chdir(str(ROOT_DIR))
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

def main():
    print(f"\n{'*'*60}")
    print("Single Model Loading Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*60}")
    
    # Test one PPO and one SAC model
    results = []
    
    test_cases = [
        ("wf_ppo_model_2024H2.zip", "ppo"),
        ("wf_sac_model_2024H2.zip", "sac"),
    ]
    
    for model_zip, algo in test_cases:
        success = test_single_model(model_zip, algo)
        results.append((model_zip, success))
    
    # Summary
    print(f"\n{'*'*60}")
    print("RESULTS")
    print(f"{'*'*60}")
    for model_zip, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_zip}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\nPassed: {success_count}/{len(results)}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
