"""Test loading models directly from zip files."""

import sys
from datetime import datetime
from pathlib import Path

# Add project to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO, SAC  # noqa: E402


def test_single_model(model_zip: str, algo: str = "ppo"):
    """Test loading a single model from zip file."""
    zip_path = ROOT_DIR / model_zip
    
    print(f"\n{'='*60}")
    print(f"Testing Model Load: {model_zip}")
    print(f"Algorithm: {algo.upper()}")
    print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    model_class = PPO if algo == "ppo" else SAC
    
    try:
        print(f"Loading from zip file: {zip_path}")
        # Load directly from zip file (not extracted directory)
        model = model_class.load(str(zip_path), device="cpu")
        
        print("\n✓ SUCCESS!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Total timesteps: {model.num_timesteps:,}")
        print(f"  Learning rate: {model.learning_rate}")
        
        # Try a quick predict if environment exists
        try:
            if model.policy is not None:
                print(f"  Policy observation space: {model.policy.observation_space}")
                print(f"  Policy action space: {model.policy.action_space}")
        except Exception:
            pass
        
        return True
        
    except Exception as e:
        print("\n✗ FAILED!")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"\n{'*'*60}")
    print("Model Loading Test (Direct from ZIP)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*60}")
    
    results = []
    
    # Test models
    test_cases = [
        ("wf_ppo_model_2024H2.zip", "ppo"),
        ("wf_ppo_model_2025H1.zip", "ppo"),
        ("wf_sac_model_2024H2.zip", "sac"),
        ("wf_sac_model_2025H1.zip", "sac"),
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
    
    # Summary by algorithm
    ppo_results = [s for m, s in results if "ppo" in m.lower()]
    sac_results = [s for m, s in results if "sac" in m.lower()]
    
    print("\nBy Algorithm:")
    print(f"  PPO: {sum(ppo_results)}/{len(ppo_results)} passed")
    print(f"  SAC: {sum(sac_results)}/{len(sac_results)} passed")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
