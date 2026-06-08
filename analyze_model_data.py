"""Analyze model data structure."""

import sys
import zipfile
from pathlib import Path

# Add project to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

def analyze_model_data(model_zip: str):
    """Analyze the data structure in a model zip."""
    zip_path = ROOT_DIR / model_zip
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_zip}")
    print(f"{'='*60}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Read data file
            data_content = z.read('data')
            
            # Try to load as JSON
            import json
            data = json.loads(data_content)
            
            print("\nModel Configuration:")
            for key, value in data.items():
                if key == 'policy_kwargs':
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, dict):
                            print(f"    {k}: {v}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            
            # Show tensorboard info if available
            print("\nModel State:")
            version = z.read('_stable_baselines3_version').decode()
            print(f"  SB3 Version: {version}")
            
            # List all files
            print("\nZip contents:")
            for file in z.namelist():
                info = z.getinfo(file)
                size_kb = info.file_size / 1024
                print(f"  {file}: {size_kb:.1f} KB")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"\n{'*'*60}")
    print("Model Data Analysis")
    print(f"{'*'*60}")
    
    models = [
        "wf_ppo_model_2024H2.zip",
        "wf_sac_model_2024H2.zip",
    ]
    
    for model in models:
        analyze_model_data(model)

if __name__ == "__main__":
    main()
