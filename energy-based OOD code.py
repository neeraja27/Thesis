#!/usr/bin/env python3
"""
Energy-based Out-of-Distribution Detection for Semantic Segmentation
Based on "Energy-based Out-of-distribution Detection" by Liu et al. NeurIPS 2020
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import json
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

class EnergyOODDetector:
    """
    Energy-based Out-of-Distribution Detection for Semantic Segmentation
    
    Fixed version that properly handles MMSegmentation model inference
    """
    
    def __init__(self, config_file, checkpoint_file, device='cuda:0', temperature=1.0):
        print(f"Initializing Energy-based OOD detector...")
        print(f"Config: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")
        print(f"Temperature: {temperature}")
        
        # Energy-based OOD parameters (define early for mock model)
        self.temperature = temperature      
        self.num_classes = 19              
        self.energy_threshold = -10.0
        
        # Try to import MMSeg components
        try:
            from mmseg.apis import init_segmentor, inference_segmentor
            import mmcv
            
            # Load configuration and model
            self.cfg = mmcv.Config.fromfile(config_file)
            self.cfg.model.pretrained = None
            self.model = init_segmentor(self.cfg, checkpoint_file, device=device)
            self.model.eval()
            
            self.device = next(self.model.parameters()).device
            print(f"Model loaded on {self.device}")
            self.use_real_model = True
            self.inference_segmentor = inference_segmentor
            
        except ImportError as e:
            print(f"Warning: MMSegmentation not available: {e}")
            print("Using mock model for demonstration")
            self._init_mock_model(device)
            self.use_real_model = False      
        
        # Fine-tuning parameters
        self.m_in = -25.0                  
        self.m_out = -7.0                  
        self.lambda_energy = 0.1           
        
        # Image preprocessing - simpler for MMSegmentation
        self.transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Storage for statistics
        self.inference_times = []
        self.energy_scores_history = []
        
        print(f"Energy OOD detector initialized with T={temperature}")
        print(f"Energy margins: ID < {self.m_in}, OOD > {self.m_out}")
    
    def _init_mock_model(self, device):
        """Initialize mock model when MMSegmentation unavailable"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        class MockSegmentationModel(nn.Module):
            def __init__(self, num_classes=19):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((64, 128)),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((32, 64)),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                )
                self.classifier = nn.Conv2d(256, num_classes, 1)
                self.upsample = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=False)
            
            def forward(self, x):
                features = self.backbone(x)
                logits = self.classifier(features)
                logits = self.upsample(logits)
                return logits
        
        self.model = MockSegmentationModel(self.num_classes).to(self.device)
        self.cfg = None
        print(f"Mock model initialized on {self.device}")
    
    def get_model_logits(self, img_path):
        """
        Get logits from MMSegmentation model using proper inference
        """
        if self.use_real_model:
            # Use MMSegmentation's built-in inference function
            # This handles all the preprocessing and metadata automatically
            result = self.inference_segmentor(self.model, img_path)
            
            # Extract logits from the model manually for energy computation
            # We need to do a forward pass to get the raw logits
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get features from backbone
                if hasattr(self.model, 'extract_feat'):
                    features = self.model.extract_feat(img_tensor)
                else:
                    features = self.model.backbone(img_tensor)
                
                # Get logits from decode head
                if hasattr(self.model, 'decode_head') and hasattr(self.model.decode_head, 'forward'):
                    logits = self.model.decode_head(features)
                else:
                    # Fallback: try to get logits another way
                    if isinstance(features, (list, tuple)):
                        features = features[-1]  # Use last feature map
                    # Add a simple classifier if needed
                    if features.shape[1] != self.num_classes:
                        if not hasattr(self, '_temp_classifier'):
                            self._temp_classifier = nn.Conv2d(
                                features.shape[1], self.num_classes, 1
                            ).to(self.device)
                        logits = self._temp_classifier(features)
                    else:
                        logits = features
                
                # Ensure correct output size
                if logits.shape[2:] != (512, 1024):
                    logits = F.interpolate(logits, size=(512, 1024), mode='bilinear', align_corners=False)
                
                return logits, result
        else:
            # Mock model
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_tensor)
                # Create mock segmentation result
                result = np.random.randint(0, self.num_classes, (512, 1024))
                return logits, result
    
    def compute_energy_score(self, logits):
        """
        Compute energy score from logits
        Energy E(x) = -T * log(sum(exp(f_i(x)/T)))
        """
        batch_size, num_classes, height, width = logits.shape
        logits_flat = logits.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Compute energy: E(x) = -T * log(sum(exp(f_i(x)/T)))
        energy_flat = -self.temperature * torch.logsumexp(logits_flat / self.temperature, dim=1)  # [B, H*W]
        energy_scores = energy_flat.view(batch_size, height, width)  # [B, H, W]
        
        return energy_scores
    
    def predict_energy_ood(self, img_path):
        """
        Predict OOD using energy score with proper MMSegmentation handling
        """
        start_time = time.time()
        
        try:
            # Get logits and segmentation result
            logits, seg_result = self.get_model_logits(img_path)
            
            # Compute energy scores
            energy_scores = self.compute_energy_score(logits)  # [1, H, W]
            energy_scores = energy_scores.squeeze(0).cpu().numpy()  # [H, W]
            
            # Compute softmax probabilities for comparison
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [C, H, W]
            probs = probs.transpose(1, 2, 0)  # [H, W, C]
            
            # Compute softmax confidence (max probability)
            softmax_conf = np.max(probs, axis=2)  # [H, W]
            
            # OOD detection using energy threshold
            ood_mask = (energy_scores > self.energy_threshold).astype(np.uint8)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Store energy scores for analysis
            mean_energy = np.mean(energy_scores)
            self.energy_scores_history.append(mean_energy)
            
            return {
                'energy_scores': energy_scores,
                'ood_mask': ood_mask,
                'softmax_confidence': softmax_conf,
                'probabilities': probs,
                'segmentation_result': seg_result,
                'mean_energy': mean_energy,
                'min_energy': np.min(energy_scores),
                'max_energy': np.max(energy_scores),
                'ood_pixel_ratio': np.mean(ood_mask),
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Error in energy OOD prediction for {img_path}: {str(e)}")
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            return None
    
    def calibrate_threshold(self, id_images, target_fpr=0.05):
        """
        Calibrate energy threshold using ID validation data
        """
        print(f"Calibrating energy threshold with {len(id_images)} ID images...")
        
        energy_scores = []
        successful_predictions = 0
        
        for i, img_path in enumerate(id_images):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(id_images)} images")
            
            result = self.predict_energy_ood(img_path)
            if result is not None:
                energy_scores.extend(result['energy_scores'].flatten())
                successful_predictions += 1
        
        if successful_predictions == 0:
            print("Warning: No successful predictions for threshold calibration")
            return self.energy_threshold
        
        energy_scores = np.array(energy_scores)
        
        # Find threshold that gives target FPR
        threshold = np.percentile(energy_scores, (1 - target_fpr) * 100)
        
        print(f"Successfully processed {successful_predictions}/{len(id_images)} images")
        print(f"Calibrated threshold: {threshold:.4f} (Target FPR: {target_fpr})")
        print(f"Energy statistics on ID data:")
        print(f"  Mean: {np.mean(energy_scores):.4f}")
        print(f"  Std: {np.std(energy_scores):.4f}")
        print(f"  Min: {np.min(energy_scores):.4f}")
        print(f"  Max: {np.max(energy_scores):.4f}")
        
        self.energy_threshold = threshold
        return threshold
    
    def evaluate_ood_detection(self, id_test_paths, ood_test_paths, id_val_paths=None):
        """
        Evaluate energy-based OOD detection
        """
        print(f"Starting Energy-based OOD evaluation...")
        print(f"ID test images: {len(id_test_paths)}")
        print(f"OOD test images: {len(ood_test_paths)}")
        
        # Calibrate threshold if validation data provided
        if id_val_paths is not None:
            self.calibrate_threshold(id_val_paths)
        
        results = {
            'id_energy_scores': [],
            'ood_energy_scores': [],
            'id_softmax_scores': [],
            'ood_softmax_scores': [],
            'processing_stats': []
        }
        
        # Process ID test images
        print("\nProcessing ID test images...")
        successful_id = 0
        for i, img_path in enumerate(id_test_paths):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(id_test_paths)} ID images")
            
            result = self.predict_energy_ood(img_path)
            if result is None:
                continue
            
            successful_id += 1
            
            # Use negative energy as score (higher = more ID-like)
            energy_score = -result['mean_energy']
            softmax_score = np.mean(result['softmax_confidence'])
            
            results['id_energy_scores'].append(energy_score)
            results['id_softmax_scores'].append(softmax_score)
            
            # Store detailed statistics
            results['processing_stats'].append({
                'type': 'ID',
                'path': img_path,
                'energy_score': energy_score,
                'softmax_score': softmax_score,
                'inference_time': result['inference_time']
            })
        
        # Process OOD test images
        print(f"\nProcessing OOD test images...")
        successful_ood = 0
        for i, img_path in enumerate(ood_test_paths):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(ood_test_paths)} OOD images")
            
            result = self.predict_energy_ood(img_path)
            if result is None:
                continue
            
            successful_ood += 1
            
            # Use negative energy as score (higher = more ID-like)
            energy_score = -result['mean_energy']
            softmax_score = np.mean(result['softmax_confidence'])
            
            results['ood_energy_scores'].append(energy_score)
            results['ood_softmax_scores'].append(softmax_score)
            
            # Store detailed statistics
            results['processing_stats'].append({
                'type': 'OOD',
                'path': img_path,
                'energy_score': energy_score,
                'softmax_score': softmax_score,
                'inference_time': result['inference_time']
            })
        
        print(f"\nSuccessfully processed:")
        print(f"  ID images: {successful_id}/{len(id_test_paths)}")
        print(f"  OOD images: {successful_ood}/{len(ood_test_paths)}")
        
        # Compute metrics if we have both ID and OOD scores
        if len(results['id_energy_scores']) > 0 and len(results['ood_energy_scores']) > 0:
            # Prepare labels and scores for evaluation
            energy_scores = np.array(results['id_energy_scores'] + results['ood_energy_scores'])
            softmax_scores = np.array(results['id_softmax_scores'] + results['ood_softmax_scores'])
            labels = np.array([0] * len(results['id_energy_scores']) + [1] * len(results['ood_energy_scores']))
            
            # Compute AUROC
            try:
                energy_auroc = roc_auc_score(labels, -energy_scores)  # negative because lower energy = more OOD
                softmax_auroc = roc_auc_score(labels, -softmax_scores)  # negative because lower confidence = more OOD
            except ValueError as e:
                print(f"Warning: Could not compute AUROC: {e}")
                energy_auroc = 0.0
                softmax_auroc = 0.0
            
            # Compute FPR95 (False Positive Rate at 95% True Positive Rate)
            def compute_fpr95(scores, labels):
                try:
                    fpr, tpr, thresholds = roc_curve(labels, scores)
                    # Find the threshold that gives TPR closest to 0.95
                    idx = np.argmin(np.abs(tpr - 0.95))
                    return fpr[idx]
                except:
                    return 1.0
            
            energy_fpr95 = compute_fpr95(-energy_scores, labels)
            softmax_fpr95 = compute_fpr95(-softmax_scores, labels)
            
            # Compute AUPR (Area Under Precision-Recall Curve)
            def compute_aupr(scores, labels):
                try:
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    return auc(recall, precision)
                except:
                    return 0.0
            
            energy_aupr = compute_aupr(-energy_scores, labels)
            softmax_aupr = compute_aupr(-softmax_scores, labels)
            
            # Store metrics in results
            results['metrics'] = {
                'energy_auroc': energy_auroc,
                'softmax_auroc': softmax_auroc,
                'energy_aupr': energy_aupr,
                'softmax_aupr': softmax_aupr,
                'energy_fpr95': energy_fpr95,
                'softmax_fpr95': softmax_fpr95,
                'auroc_improvement': energy_auroc - softmax_auroc,
                'aupr_improvement': energy_aupr - softmax_aupr,
                'fpr95_improvement': softmax_fpr95 - energy_fpr95,
                'energy_separation': np.mean(results['ood_energy_scores']) - np.mean(results['id_energy_scores']),
                'avg_inference_time': np.mean([stat['inference_time'] for stat in results['processing_stats']])
            }
            
            print(f"\n=== Energy-based OOD Detection Results ===")
            print(f"Energy AUROC: {energy_auroc:.4f}")
            print(f"Softmax AUROC: {softmax_auroc:.4f}")
            print(f"AUROC improvement: {energy_auroc - softmax_auroc:.4f}")
            print(f"Energy AUPR: {energy_aupr:.4f}")
            print(f"Softmax AUPR: {softmax_aupr:.4f}")
            print(f"AUPR improvement: {energy_aupr - softmax_aupr:.4f}")
            print(f"Energy FPR95: {energy_fpr95:.4f}")
            print(f"Softmax FPR95: {softmax_fpr95:.4f}")
            print(f"FPR95 improvement: {softmax_fpr95 - energy_fpr95:.4f}")
            print(f"Energy separation (OOD - ID): {np.mean(results['ood_energy_scores']) - np.mean(results['id_energy_scores']):.4f}")
            print(f"Average inference time: {np.mean([stat['inference_time'] for stat in results['processing_stats']]):.4f}s")
        
        return results
    
    def plot_results(self, results, save_path='energy_ood_plots.png'):
        """
        Plot energy-based OOD detection results
        Fixed to handle the correct dictionary keys
        """
        # Check if we have the required data
        if ('id_energy_scores' not in results or 'ood_energy_scores' not in results or
            len(results['id_energy_scores']) == 0 or len(results['ood_energy_scores']) == 0):
            print("Warning: Insufficient data for plotting. Skipping plot generation.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy-based OOD Detection Results', fontsize=16)
        
        # Extract data
        id_energy = np.array(results['id_energy_scores'])
        ood_energy = np.array(results['ood_energy_scores'])
        id_softmax = np.array(results['id_softmax_scores'])
        ood_softmax = np.array(results['ood_softmax_scores'])
        
        # Plot 1: Energy score distributions
        axes[0, 0].hist(id_energy, bins=50, alpha=0.7, label='ID', density=True, color='blue')
        axes[0, 0].hist(ood_energy, bins=50, alpha=0.7, label='OOD', density=True, color='red')
        axes[0, 0].set_xlabel('Energy Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Energy Score Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Softmax confidence distributions
        axes[0, 1].hist(id_softmax, bins=50, alpha=0.7, label='ID', density=True, color='blue')
        axes[0, 1].hist(ood_softmax, bins=50, alpha=0.7, label='OOD', density=True, color='red')
        axes[0, 1].set_xlabel('Softmax Confidence')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Softmax Confidence Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ROC curves
        if 'metrics' in results:
            # Prepare data for ROC curves
            all_energy = np.concatenate([id_energy, ood_energy])
            all_softmax = np.concatenate([id_softmax, ood_softmax])
            labels = np.concatenate([np.zeros(len(id_energy)), np.ones(len(ood_energy))])
            
            try:
                from sklearn.metrics import roc_curve, precision_recall_curve
                
                # Energy ROC
                fpr_energy, tpr_energy, _ = roc_curve(labels, -all_energy)
                axes[1, 0].plot(fpr_energy, tpr_energy, 
                               label=f'Energy (AUROC: {results["metrics"]["energy_auroc"]:.3f})', 
                               linewidth=2, color='blue')
                
                # Softmax ROC
                fpr_softmax, tpr_softmax, _ = roc_curve(labels, -all_softmax)
                axes[1, 0].plot(fpr_softmax, tpr_softmax, 
                               label=f'Softmax (AUROC: {results["metrics"]["softmax_auroc"]:.3f})', 
                               linewidth=2, color='red')
                
                axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curves')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Precision-Recall curves
                # Energy PR curve
                precision_energy, recall_energy, _ = precision_recall_curve(labels, -all_energy)
                axes[1, 1].plot(recall_energy, precision_energy,
                               label=f'Energy (AUPR: {results["metrics"]["energy_aupr"]:.3f})',
                               linewidth=2, color='blue')
                
                # Softmax PR curve
                precision_softmax, recall_softmax, _ = precision_recall_curve(labels, -all_softmax)
                axes[1, 1].plot(recall_softmax, precision_softmax,
                               label=f'Softmax (AUPR: {results["metrics"]["softmax_aupr"]:.3f})',
                               linewidth=2, color='red')
                
                axes[1, 1].set_xlabel('Recall')
                axes[1, 1].set_ylabel('Precision')
                axes[1, 1].set_title('Precision-Recall Curves')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'ROC curve error:\n{str(e)}', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('ROC Curves (Error)')
                axes[1, 1].text(0.5, 0.5, f'PR curve error:\n{str(e)}',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('PR Curves (Error)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_path}")
    
    def energy_regularized_training(self, model, train_loader, val_loader, num_epochs=10, lr=1e-4):
        """
        Energy-regularized training for improved OOD detection
        """
        print(f"Starting energy-regularized training for {num_epochs} epochs...")
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training metrics
        training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'energy_loss': [],
            'ce_loss': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            total_loss = 0.0
            total_ce_loss = 0.0
            total_energy_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(images)
                
                # Standard cross-entropy loss
                ce_loss = F.cross_entropy(logits, targets, ignore_index=255)
                
                # Energy regularization
                energy_scores = self.compute_energy_score(logits)
                
                # Energy regularization: encourage low energy for ID data
                energy_loss = torch.mean(torch.clamp(energy_scores - self.m_in, min=0))
                
                # Total loss
                total_batch_loss = ce_loss + self.lambda_energy * energy_loss
                
                # Backward pass
                total_batch_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                total_loss += total_batch_loss.item()
                total_ce_loss += ce_loss.item()
                total_energy_loss += energy_loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx+1}: Loss={total_batch_loss.item():.4f}, "
                          f"CE={ce_loss.item():.4f}, Energy={energy_loss.item():.4f}")
            
            # Average training losses
            avg_train_loss = total_loss / num_batches
            avg_ce_loss = total_ce_loss / num_batches
            avg_energy_loss = total_energy_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets, ignore_index=255)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            
            # Store training history
            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['energy_loss'].append(avg_energy_loss)
            training_history['ce_loss'].append(avg_ce_loss)
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  CE Loss: {avg_ce_loss:.4f}, Energy Loss: {avg_energy_loss:.4f}")
        
        print("Energy-regularized training completed!")
        return training_history
    
    def save_results(self, results, filename='energy_ood_results.json'):
        """
        Save evaluation results to JSON file
        """
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types to JSON serializable types
        serializable_results = convert_numpy_types(results)
        
        # Add timestamp and parameters
        serializable_results['timestamp'] = datetime.now().isoformat()
        serializable_results['parameters'] = {
            'temperature': float(self.temperature),
            'energy_threshold': float(self.energy_threshold),
            'num_classes': int(self.num_classes),
            'm_in': float(self.m_in),
            'm_out': float(self.m_out),
            'lambda_energy': float(self.lambda_energy)
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")

def get_image_paths(directory, extensions=['.png', '.jpg', '.jpeg', 'webp']):
    """Get all image paths from directory"""
    import glob
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, '**', f'*{ext}')
        paths = glob.glob(pattern, recursive=True)
        image_paths.extend(paths)
        pattern = os.path.join(directory, '**', f'*{ext.upper()}')
        paths = glob.glob(pattern, recursive=True)
        image_paths.extend(paths)
    return sorted(list(set(image_paths)))


# Example usage
if __name__ == "__main__":
    # Configuration
    config_file = "configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py"
    checkpoint_file = "checkpoints/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.pth"
    device = 'cuda:0'
    temperature = 1.0
    
    # Dataset paths
    training_dir = "newdata/cityscape_newsubset/leftImg8bit"
    ood_dir = "newdata/smiyc"
    
    # Get image paths
    all_training_paths = get_image_paths(training_dir)
    ood_paths = get_image_paths(ood_dir)
    
    print(f"Found {len(all_training_paths)} images in training directory")
    print(f"Found {len(ood_paths)} images in OOD directory")
    
    # Split data
    np.random.shuffle(all_training_paths)
    val_split_idx = int(len(all_training_paths) * 0.3)  
    test_split_idx = int(len(all_training_paths) * 0.7)  
    
    id_val_paths = all_training_paths[:val_split_idx]
    id_test_paths = all_training_paths[val_split_idx:test_split_idx]
    
    print(f"\nDataset split:")
    print(f"  ID Validation: {len(id_val_paths)} images")
    print(f"  ID Test: {len(id_test_paths)} images")
    print(f"  OOD Test: {len(ood_paths)} images")
    
    # Initialize detector and run evaluation
    detector = EnergyOODDetector(config_file, checkpoint_file, device, temperature)
    results = detector.evaluate_ood_detection(id_test_paths, ood_paths, id_val_paths)
    
    if results is not None:
        # Save results and plots
        detector.save_results(results, 'energy_ood_results.json')
        detector.plot_results(results, 'energy_ood_plots.png')
    else:
        print("Evaluation failed - no results to save")