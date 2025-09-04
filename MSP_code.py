#!/usr/bin/env python3
"""
MSP OOD Detection using MMSegmentation
"""
import time
import torch
import numpy as np
from mmseg.apis import init_segmentor
import mmcv
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import json
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms

class SimpleMSPDetector:
    """Simplified MSP detector that avoids DataContainer complications"""
    
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        print(f"Initializing simplified MSP detector...")
        print(f"Config: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")
        
        # Load configuration and model
        self.cfg = mmcv.Config.fromfile(config_file)
        self.cfg.model.pretrained = None
        self.model = init_segmentor(self.cfg, checkpoint_file, device=device)
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        
        # Define image preprocessing transforms (similar to MMSeg pipeline)
        self.transform = transforms.Compose([
            transforms.Resize((512, 1024)),  # Standard Cityscapes size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.inference_times = []
    
    def preprocess_image(self, img_path):
        """Simple image preprocessing without MMSeg pipeline complications"""
        # Load image with PIL
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Create simple metadata (required by model)
        img_meta = {
            'filename': img_path,
            'ori_shape': (img.height, img.width, 3),
            'img_shape': (512, 1024, 3),
            'pad_shape': (512, 1024, 3),
            'scale_factor': np.array([1024/img.width, 512/img.height, 1024/img.width, 512/img.height]),
            'flip': False,
            'flip_direction': None,
            'img_norm_cfg': {
                'mean': np.array([0.485, 0.456, 0.406]) * 255,
                'std': np.array([0.229, 0.224, 0.225]) * 255,
                'to_rgb': True
            }
        }
        
        return img_tensor, img_meta
    
    def compute_msp_score(self, img_path):
        """Compute MSP score using direct model forward pass"""
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Preprocess image
                img_tensor, img_meta = self.preprocess_image(img_path)
                
                # Direct forward pass through the model
                # Use the model's simple_test method which handles the forward pass
                result = self.model.simple_test(img_tensor, [img_meta], rescale=False)
                
                # Get logits directly by calling encode_decode
                logits = self.model.encode_decode(img_tensor, [img_meta])
                
                # Calculate softmax probabilities
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits).to(self.device)
                
                probs = torch.softmax(logits, dim=1)
                
                # Get maximum softmax probability per pixel
                max_probs = probs.max(dim=1)[0]  # Shape: [H, W]
                
                # Calculate mean MSP score across all pixels
                msp_score = max_probs.mean().cpu().numpy()
                
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                return float(msp_score)
                
        except Exception as e:
            print(f"Primary method failed for {img_path}: {str(e)}")
            # Try alternative approach
            try:
                return self._compute_msp_alternative(img_path)
            except Exception as e2:
                print(f"Alternative method failed for {img_path}: {str(e2)}")
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                return 0.5  # Neutral score
    
    def _compute_msp_alternative(self, img_path):
        """Alternative method using model's feature extraction"""
        with torch.no_grad():
            img_tensor, img_meta = self.preprocess_image(img_path)
            
            # Extract features
            if hasattr(self.model, 'extract_feat'):
                features = self.model.extract_feat(img_tensor)
            else:
                # Use backbone directly
                features = self.model.backbone(img_tensor)
            
            # Use decode head to get logits
            if hasattr(self.model, 'decode_head'):
                logits = self.model.decode_head(features)
            else:
                raise ValueError("Model has no decode_head")
            
            # Process logits
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits).to(self.device)
            
            # Calculate MSP
            probs = torch.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            msp_score = max_probs.mean().cpu().numpy()
            
            return float(msp_score)
    
    def evaluate_ood_detection(self, in_dist_paths, ood_paths, max_samples=None):
        """Run complete OOD evaluation"""
        print(f"Starting simplified MSP evaluation...")
        
        # Use all images if max_samples is None
        if max_samples is None:
            id_count = len(in_dist_paths)
            ood_count = len(ood_paths)
        else:
            id_count = min(len(in_dist_paths), max_samples)
            ood_count = min(len(ood_paths), max_samples)
            
        print(f"Processing {id_count} ID images from {len(in_dist_paths)} available")
        print(f"Processing {ood_count} OOD images from {len(ood_paths)} available")
        
        in_dist_scores = []
        ood_scores = []
        
        # Process in-distribution images
        print("\nProcessing in-distribution images...")
        id_paths_to_process = in_dist_paths[:max_samples] if max_samples else in_dist_paths
        
        for i, img_path in enumerate(id_paths_to_process):
            msp_score = self.compute_msp_score(img_path)
            in_dist_scores.append(msp_score)
            
            # Show progress every 10 images for larger datasets
            progress_interval = max(1, len(id_paths_to_process) // 10)
            if (i + 1) % progress_interval == 0 or i == len(id_paths_to_process) - 1:
                print(f"  ID {i+1}/{len(id_paths_to_process)}: MSP = {msp_score:.4f}, "
                      f"Avg time: {np.mean(self.inference_times[-10:]):.3f}s")
        
        # Process out-of-distribution images
        print("\nProcessing out-of-distribution images...")
        ood_paths_to_process = ood_paths[:max_samples] if max_samples else ood_paths
        
        for i, img_path in enumerate(ood_paths_to_process):
            msp_score = self.compute_msp_score(img_path)
            ood_scores.append(msp_score)
            
            # Show progress every 10 images for larger datasets
            progress_interval = max(1, len(ood_paths_to_process) // 10)
            if (i + 1) % progress_interval == 0 or i == len(ood_paths_to_process) - 1:
                print(f"  OOD {i+1}/{len(ood_paths_to_process)}: MSP = {msp_score:.4f}, "
                      f"Avg time: {np.mean(self.inference_times[-10:]):.3f}s")
        
        # Calculate metrics
        return self._calculate_metrics(np.array(in_dist_scores), np.array(ood_scores))
    
    def _calculate_metrics(self, in_dist_scores, ood_scores):
        """Calculate evaluation metrics"""
        # Combine scores and labels
        all_scores = np.concatenate([in_dist_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
        
        # Calculate AUROC (negative scores since lower MSP = OOD)
        auroc = roc_auc_score(labels, -all_scores)
        
        # Calculate FPR at different TPR levels
        fpr, tpr, thresholds = roc_curve(labels, -all_scores)
        
        # Calculate Precision-Recall metrics
        precision, recall, _ = precision_recall_curve(labels, -all_scores)
        aupr = average_precision_score(labels, -all_scores)
        
        def get_fpr_at_tpr(target_tpr):
            idx = np.argmax(tpr >= target_tpr)
            return fpr[idx] if np.any(tpr >= target_tpr) else 1.0
        
        metrics = {
            'auroc': float(auroc),
            'aupr': float(aupr),  # Added AUPR metric
            'fpr95': float(get_fpr_at_tpr(0.95)),
            'fpr90': float(get_fpr_at_tpr(0.90)),
            'fpr85': float(get_fpr_at_tpr(0.85)),
            'avg_inference_time': float(np.mean(self.inference_times)),
            'total_inference_time': float(np.sum(self.inference_times)),
            'in_dist_stats': {
                'mean': float(np.mean(in_dist_scores)),
                'std': float(np.std(in_dist_scores)),
                'count': len(in_dist_scores)
            },
            'ood_stats': {
                'mean': float(np.mean(ood_scores)),
                'std': float(np.std(ood_scores)),
                'count': len(ood_scores)
            },
            'separation': float(np.mean(in_dist_scores) - np.mean(ood_scores)),
            'roc_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'pr_data': {  # Added PR curve data
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        }
        
        return metrics, in_dist_scores, ood_scores
    
    def save_results(self, metrics, in_dist_scores, ood_scores, output_dir='results'):
        """Save results with visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"simple_msp_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Score distributions
        ax1.hist(in_dist_scores, bins=20, alpha=0.7, label='In-Distribution', density=True, color='blue')
        ax1.hist(ood_scores, bins=20, alpha=0.7, label='Out-of-Distribution', density=True, color='red')
        ax1.set_xlabel('MSP Score')
        ax1.set_ylabel('Density')
        ax1.set_title('MSP Score Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ROC curve
        fpr = np.array(metrics['roc_data']['fpr'])
        tpr = np.array(metrics['roc_data']['tpr'])
        ax2.plot(fpr, tpr, linewidth=2, label=f'AUROC = {metrics["auroc"]:.4f}', color='blue')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PR curve (new plot)
        precision = np.array(metrics['pr_data']['precision'])
        recall = np.array(metrics['pr_data']['recall'])
        ax3.plot(recall, precision, linewidth=2, label=f'AUPR = {metrics["aupr"]:.4f}', color='green')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"simple_msp_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Plots: {plot_path}")
        
        return json_path, plot_path

def get_image_paths(directory, max_images=None):
    """Get image paths from directory"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    extensions = ('.png', '.jpg', '.jpeg','webp')
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    
    if not image_files:
        print(f"No images found in {directory}")
        return []
    
    image_files.sort()  # For consistent ordering
    
    if max_images:
        paths = [os.path.join(directory, f) for f in image_files[:max_images]]
    else:
        paths = [os.path.join(directory, f) for f in image_files]
    
    print(f"Found {len(paths)} images in {directory}")
    return paths

def main():
    """Main evaluation function"""
    print("="*70)
    print("Simplified MSP OOD Detection with MMSegmentation")
    print("="*70)
    
    # Configuration
    config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
    checkpoint_file = 'checkpoints/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.pth'
    
    # Verify files exist
    for file_path in [config_file, checkpoint_file]:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    # Dataset directories
    in_dist_dir = 'newdata/cityscape_newsubset/leftImg8bit'
    ood_dir = 'newdata/smiyc'
    
    # Get all available image paths (no limit)
    in_dist_paths = get_image_paths(in_dist_dir)  # Load all Cityscapes images
    ood_paths = get_image_paths(ood_dir)          # Load all Lost & Found images
    
    if not in_dist_paths or not ood_paths:
        print("Error: Insufficient images found in directories")
        return
    
    try:
        # Initialize detector
        detector = SimpleMSPDetector(config_file, checkpoint_file)
        
        # Run evaluation on full dataset (remove max_samples limit)
        metrics, in_dist_scores, ood_scores = detector.evaluate_ood_detection(
            in_dist_paths, ood_paths, max_samples=None  # Use all available images
        )
        
        # Display results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPR: {metrics['aupr']:.4f}")  # Added AUPR display
        print(f"FPR at TPR 95%: {metrics['fpr95']:.4f}")
        print(f"FPR at TPR 90%: {metrics['fpr90']:.4f}")
        print(f"FPR at TPR 85%: {metrics['fpr85']:.4f}")
        print(f"Average inference time: {metrics['avg_inference_time']:.4f}s")
        
        print(f"\nScore Statistics:")
        print(f"  In-Distribution:  {metrics['in_dist_stats']['mean']:.4f} ± {metrics['in_dist_stats']['std']:.4f}")
        print(f"  Out-Distribution: {metrics['ood_stats']['mean']:.4f} ± {metrics['ood_stats']['std']:.4f}")
        print(f"  Separation: {metrics['separation']:.4f}")
        
        # Performance assessment
        if metrics['auroc'] >= 0.9 and metrics['aupr'] >= 0.9:
            assessment = "Excellent"
        elif metrics['auroc'] >= 0.8 and metrics['aupr'] >= 0.8:
            assessment = "Good"
        elif metrics['auroc'] >= 0.7 or metrics['aupr'] >= 0.7:
            assessment = "Moderate"
        else:
            assessment = "Poor"
        
        print(f"\nPerformance: {assessment}")
        
        # Save results
        detector.save_results(metrics, in_dist_scores, ood_scores)
        
        print("="*70)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    main()
