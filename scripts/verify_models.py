"""Verify zip model files directly."""

import zipfile
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

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

def verify_zip(zip_path: Path) -> dict:
    """Verify a zip file is valid and contains model files."""
    model_name = zip_path.stem
    algo = "ppo" if "ppo" in model_name.lower() else "sac"
    
    print(f"\n{'='*60}")
    print(f"Verifying: {model_name}")
    print(f"File size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = z.namelist()
            print(f"✓ Valid zip file with {len(files)} files:")
            
            # Check for essential files
            has_data = any('data' in f for f in files)
            has_policy = any('policy' in f for f in files)
            has_torch = any('.pth' in f for f in files)
            
            for f in sorted(files):
                print(f"  - {f}")
            
            print("\nEssential files:")
            print(f"  data file: {'✓' if has_data else '✗'}")
            print(f"  policy file: {'✓' if has_policy else '✗'}")
            print(f"  torch weights: {'✓' if has_torch else '✗'}")
            
            return {
                "status": "valid",
                "model_name": model_name,
                "algo": algo,
                "file_count": len(files),
                "has_data": has_data,
                "has_policy": has_policy,
                "has_torch": has_torch,
            }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "status": "invalid",
            "model_name": model_name,
            "error": str(e),
        }

def main():
    print(f"\n{'*'*60}")
    print("Model ZIP File Verification")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*60}")
    
    results = []
    
    for model_file in MODEL_FILES:
        zip_path = ROOT_DIR / model_file
        if not zip_path.exists():
            print(f"\n✗ File not found: {model_file}")
            results.append({
                "status": "missing",
                "model_name": model_file,
            })
            continue
        
        result = verify_zip(zip_path)
        results.append(result)
    
    # Summary
    print(f"\n{'*'*60}")
    print("SUMMARY")
    print(f"{'*'*60}")
    
    valid = sum(1 for r in results if r["status"] == "valid")
    invalid = sum(1 for r in results if r["status"] == "invalid")
    missing = sum(1 for r in results if r["status"] == "missing")
    
    print(f"Total models: {len(MODEL_FILES)}")
    print(f"✓ Valid: {valid}")
    print(f"✗ Invalid: {invalid}")
    print(f"? Missing: {missing}")
    
    # PPO/SAC breakdown
    ppo_valid = sum(1 for r in results if r.get("algo") == "ppo" and r["status"] == "valid")
    sac_valid = sum(1 for r in results if r.get("algo") == "sac" and r["status"] == "valid")
    
    print("\nBy Algorithm:")
    print(f"  PPO: {ppo_valid}/4")
    print(f"  SAC: {sac_valid}/4")
    
    print("\nDetailed Results:")
    for r in results:
        status_icon = "✓" if r["status"] == "valid" else "✗" if r["status"] == "invalid" else "?"
        print(f"  {status_icon} {r['model_name']}: {r['status']}")
        if r["status"] == "valid":
            print(f"      Files: {r['file_count']}, Data: {'✓' if r['has_data'] else '✗'}, "
                  f"Policy: {'✓' if r['has_policy'] else '✗'}, Torch: {'✓' if r['has_torch'] else '✗'}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
