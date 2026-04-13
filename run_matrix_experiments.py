import os
import subprocess
import argparse

DATASETS = ['cifar100', 'svhn', 'cifar10n']
MODELS = ['simple_cnn', 'resnet', 'mlp']
STRATEGIES = ['baseline', 'pseudo_labeling', 'consistency', 'co_training', 'adas_wsl']

def run_experiments(dry_run=False, fast_dev_run=False):
    epochs = 2 if fast_dev_run else 35
    total_runs = len(DATASETS) * len(MODELS) * len(STRATEGIES)
    current_run = 0
    
    print(f"Starting experiment matrix: {total_runs} total runs projected.")
    if dry_run:
        print("DRY RUN MODE: Commands will only be printed, not executed.")
        
    for dataset in DATASETS:
        for model in MODELS:
            for strategy in STRATEGIES:
                current_run += 1
                cmd = [
                    "python", "src/train.py",
                    "--dataset", dataset,
                    "--model_type", model,
                    "--strategy", strategy,
                    "--epochs", str(epochs),
                    "--batch_size", "128"
                ]
                
                print(f"[{current_run}/{total_runs}] Running: {' '.join(cmd)}")
                if not dry_run:
                    # Provide pythonpath via env for subprocess
                    env = os.environ.copy()
                    
                    # Cross-platform PYTHONPATH mapping (fixes Linux server crash)
                    current_dir = os.path.abspath(os.getcwd())
                    src_dir = os.path.join(current_dir, "src")
                    env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{src_dir}"
                    
                    try:
                        subprocess.run(cmd, env=env, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing run {current_run}: {e}. Continuing to next...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run with 2 epochs for quick verification")
    args = parser.parse_args()
    
    run_experiments(args.dry_run, args.fast_dev_run)
