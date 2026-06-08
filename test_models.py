"""Test all zip models to verify they load and run correctly."""

import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, SAC

ROOT_DIR = Path(__file__).resolve().parent

# Model files to test
MODEL_FILES = [
    "wf_ppo_model_2024H2.zip",
    "wf_ppo_model_2025H1.zip",
    "wf_ppo_model_2025H2.zip",
    "wf_ppo_model_2026H1.zip",
    "wf_sac_model_2024H2.zip",
    "wf_sac_model_2025H1.zip",
    "wf_sac_model_2025H2.zip",
    "wf_sac_model_2026H1.zip",
]


def extract_and_test_model(zip_path: Path, temp_dir: Path) -> dict:
    """Extract and test a model zip file."""
    model_name = zip_path.stem
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    # Determine algorithm from filename
    algo = "ppo" if "ppo" in model_name.lower() else "sac"
    model_class = PPO if algo == "ppo" else SAC
    
    # Create extraction directory in temp
    extract_dir = temp_dir / model_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract zip
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # List extracted files
        extracted_files = list(extract_dir.rglob("*"))
        print(f"Extracted {len([f for f in extracted_files if f.is_file()])} files")
        for f in sorted(extracted_files)[:5]:
            if f.is_file():
                print(f"  - {f.name}")
        
        # Load model (PPO/SAC model load expects the directory containing 'data' file)
        print(f"Loading {algo.upper()} model...")
        model = model_class.load(str(extract_dir), device="cpu")
        print("✓ Model loaded successfully")
        print(f"  Policy: {type(model.policy).__name__}")
        print(f"  Total timesteps: {model.num_timesteps}")
        
        return {
            "status": "success",
            "model_name": model_name,
            "algo": algo,
            "timesteps": model.num_timesteps,
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "status": "error",
            "model_name": model_name,
            "error": str(e),
        }


def main():
    print(f"\n{'*'*60}")
    print("Model Testing Report")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*60}")
    
    results = []
    
    # Use temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for model_file in MODEL_FILES:
            zip_path = ROOT_DIR / model_file
            if not zip_path.exists():
                print(f"\n✗ File not found: {model_file}")
                results.append({
                    "status": "missing",
                    "model_name": model_file,
                })
                continue
            
            result = extract_and_test_model(zip_path, temp_path)
            results.append(result)
    
    # Summary
    print(f"\n{'*'*60}")
    print("SUMMARY")
    print(f"{'*'*60}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    missing_count = sum(1 for r in results if r["status"] == "missing")
    
    print(f"Total models: {len(MODEL_FILES)}")
    print(f"✓ Success: {success_count}")
    print(f"✗ Errors: {error_count}")
    print(f"? Missing: {missing_count}")
    
    print("\nDetailed Results:")
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗" if r["status"] == "error" else "?"
        print(f"  {status_icon} {r['model_name']}: {r['status']}")
        if r["status"] == "success":
            print(f"      Algorithm: {r['algo'].upper()}, Timesteps: {r['timesteps']}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
