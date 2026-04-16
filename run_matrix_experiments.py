import os
import subprocess
import argparse

DATASETS = ['cifar100', 'svhn', 'cifar10n']
MODELS = ['simple_cnn', 'resnet', 'mlp']
STRATEGIES = ['baseline', 'pseudo_labeling', 'consistency', 'co_training', 'adas_wsl']

EPOCH_CONFIGS = [
    (35, "matrix_results_35epochs"),
    (50, "matrix_results_50epochs"),
    (80, "matrix_results_80epochs"),
]

def run_experiments(dry_run=False, fast_dev_run=False):
    if fast_dev_run:
        epoch_configs = [(2, "matrix_results_dev")]
    else:
        epoch_configs = EPOCH_CONFIGS

    total_runs = len(epoch_configs) * len(DATASETS) * len(MODELS) * len(STRATEGIES)
    current_run = 0

    print(f"Starting experiment matrix: {total_runs} total runs projected.")
    if dry_run:
        print("DRY RUN MODE: Commands will only be printed, not executed.")

    for epochs, output_dir in epoch_configs:
        print(f"\n{'='*60}")
        print(f"  Starting {epochs}-epoch runs → {output_dir}/")
        print(f"{'='*60}")

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
                        "--batch_size", "128",
                        "--output_dir", output_dir
                    ]

                    print(f"[{current_run}/{total_runs}] Running: {' '.join(cmd)}")
                    if not dry_run:
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
