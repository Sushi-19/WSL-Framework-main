import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents hang on Windows
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.models.unified_wsl import UnifiedWSLModel
from sklearn.metrics import confusion_matrix, classification_report

class BenchmarkEvaluator:
    """Evaluator for benchmarking WSL methods"""
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
    
    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate a single model"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle different model types
                if isinstance(model, UnifiedWSLModel):
                    # For WSL models, use the base model for evaluation
                    outputs = model.base_model(inputs)
                else:
                    # For regular models, use forward pass
                    outputs = model(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted'),
            'recall': recall_score(all_targets, all_preds, average='weighted'),
            'f1': f1_score(all_targets, all_preds, average='weighted')
        }
        
        return metrics
    
    def compare_methods(
        self,
        models: Dict[str, nn.Module],
        dataset_name: str
    ) -> pd.DataFrame:
        """Compare multiple methods"""
        results = []
        
        for name, model in tqdm(models.items(), desc="Comparing methods"):
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate_model(model)
            metrics['method'] = name
            metrics['dataset'] = dataset_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str,
        save_path: Optional[str] = None
    ):
        """Plot comparison of methods"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='method', y=metric)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Generate evaluation report"""
        report = "# Model Evaluation Report\n\n"
        
        # Overall metrics
        report += "## Overall Metrics\n\n"
        report += results_df.to_markdown(index=False)
        report += "\n\n"
        
        # Best method for each metric
        report += "## Best Methods\n\n"
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_method = results_df.loc[results_df[metric].idxmax()]
            report += f"- Best {metric}: {best_method['method']} ({best_method[metric]:.4f})\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

class AblationStudy:
    """Study the impact of different components"""
    def __init__(
        self,
        base_model: nn.Module,
        test_loader: DataLoader,
        device: torch.device
    ):
        self.base_model = base_model
        self.test_loader = test_loader
        self.device = device
        self.evaluator = BenchmarkEvaluator(base_model, test_loader, device)
    
    def study_components(
        self,
        components: Dict[str, object],
        component_name: str
    ) -> pd.DataFrame:
        """Study impact of different components"""
        results = []
        
        # Evaluate base model
        print("\nEvaluating base model...")
        base_metrics = self.evaluator.evaluate_model(self.base_model)
        base_metrics[component_name] = 'none'
        results.append(base_metrics)
        
        # Evaluate with each component
        for name, component in tqdm(components.items(), desc="Studying components"):
            print(f"\nEvaluating with {name}...")
            # Create model with component
            model = self._create_model_with_component(component)
            
            # Evaluate
            metrics = self.evaluator.evaluate_model(model)
            metrics[component_name] = name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _create_model_with_component(self, component: object) -> nn.Module:
        """Create model with a single component"""
        # This is a placeholder - implement based on your model architecture
        return self.base_model
    
    def plot_ablation(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot ablation study results"""
        plt.figure(figsize=(12, 6))
        
        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            sns.barplot(data=results_df, x=results_df.columns[-1], y=metric)
            plt.title(f'{metric.capitalize()}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class ErrorAnalysis:
    """Analyze model errors"""
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: List[str]
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
    
    def analyze_errors(self) -> Dict[str, pd.DataFrame]:
        """Analyze different types of errors"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Analyzing errors"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle different model types
                if isinstance(self.model, UnifiedWSLModel):
                    # For WSL models, use the base model for evaluation
                    outputs = self.model.base_model(inputs)
                else:
                    # For regular models, use forward pass
                    outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Analyze different types of errors
        results = {}
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        results['confusion_matrix'] = pd.DataFrame(
            cm,
            index=self.class_names,
            columns=self.class_names
        )
        
        # Classification report
        report = classification_report(
            all_targets,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )
        results['classification_report'] = pd.DataFrame(report).transpose()
        
        # Error cases
        errors = all_preds != all_targets
        error_probs = all_probs[errors]
        error_preds = all_preds[errors]
        error_targets = all_targets[errors]
        
        results['error_cases'] = pd.DataFrame({
            'true_class': [self.class_names[t] for t in error_targets],
            'predicted_class': [self.class_names[p] for p in error_preds],
            'confidence': np.max(error_probs, axis=1)
        })
        
        return results
    
    def plot_error_analysis(
        self,
        results: Dict[str, pd.DataFrame],
        save_dir: Optional[str] = None
    ):
        """Plot error analysis results"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        if save_dir:
            plt.savefig(f'{save_dir}/confusion_matrix.png')
        plt.close()
        
        # Plot error cases
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=results['error_cases'],
            x='true_class',
            y='confidence'
        )
        plt.title('Confidence Distribution for Errors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/error_confidence.png')
        plt.close()


if __name__ == '__main__':
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.models.baseline import SimpleCNN, ResNet, MLP
    from src.data.dataset_loader import get_datasets
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Run benchmark evaluation on a trained model')
    parser.add_argument('--dataset',    type=str, default='cifar100',
                        choices=['cifar100', 'cifar10n', 'svhn'],
                        help='Dataset to evaluate on')
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'resnet', 'mlp'],
                        help='Model architecture')
    parser.add_argument('--strategy',   type=str, default='adas_wsl',
                        help='Strategy name (used to locate checkpoint folder)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to best_model.pt (auto-detected if not set)')
    parser.add_argument('--data_dir',   type=str, default='./data',
                        help='Directory to download/load datasets')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Resolve checkpoint path ────────────────────────────────────────────────
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            'experiments', 'matrix_results',
            f"{args.dataset}_{args.model_type}_{args.strategy}",
            'best_model.pt'
        )

    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints are in experiments/matrix_results/")
        sys.exit(1)

    # ── Dataset config ─────────────────────────────────────────────────────────
    NUM_CLASSES = {'cifar100': 100, 'cifar10n': 10, 'svhn': 10}
    num_classes = NUM_CLASSES[args.dataset]

    CLASS_NAMES = {
        'cifar100': [
            'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
            'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
            'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
            'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
            'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
            'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
            'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
            'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
            'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
            'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
        ],
        'cifar10n': ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
        'svhn':     [str(i) for i in range(10)]
    }
    class_names = CLASS_NAMES[args.dataset]

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset} test set...")
    _, test_dataset = get_datasets(args.dataset, data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # ── Build model ────────────────────────────────────────────────────────────
    if args.model_type == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    elif args.model_type == 'resnet':
        model = ResNet(num_classes=num_classes, in_channels=3)
    else:
        model = MLP(input_size=3 * 32 * 32, num_classes=num_classes)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip 'model1.' prefix if checkpoint was saved from a UnifiedWSLModel wrapper
    if any(k.startswith('model1.') for k in state.keys()):
        print("  Detected 'model1.' prefix — stripping wrapper keys...")
        state = {k.replace('model1.', '', 1): v for k, v in state.items()
                 if k.startswith('model1.')}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in checkpoint")
    model.to(device).eval()

    # ── BenchmarkEvaluator ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK EVALUATION")
    print(f"  Dataset: {args.dataset} | Model: {args.model_type} | Strategy: {args.strategy}")
    print(f"{'='*60}")

    evaluator = BenchmarkEvaluator(model, test_loader, device)
    metrics   = evaluator.evaluate_model(model)

    print(f"\nResults:")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1-Score  : {metrics['f1']*100:.2f}%")

    # ── ErrorAnalysis ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS")
    print(f"{'='*60}")

    error_analyzer = ErrorAnalysis(model, test_loader, device, class_names)
    error_results  = error_analyzer.analyze_errors()

    print("\nClassification Report (top 10 classes):")
    print(error_results['classification_report'].head(10).to_string())

    print(f"\nTop 10 most common error pairs:")
    error_df = error_results['error_cases']
    top_errors = (error_df.groupby(['true_class', 'predicted_class'])
                  .size().reset_index(name='count')
                  .sort_values('count', ascending=False).head(10))
    print(top_errors.to_string(index=False))

    print(f"\nTotal errors: {len(error_df)} / {len(test_dataset)}")
    print(f"Error rate  : {len(error_df)/len(test_dataset)*100:.2f}%")