import os
import time
import gc
import json
import glob
import mmcv
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmseg.apis import init_segmentor
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.covariance import LedoitWolf  # shrinkage covariance
from datetime import datetime
from typing import List, Tuple


class MahalanobisDetector:
    def __init__(self, config_file, checkpoint_file, device='cuda:1', eps=1e-6):
        """
        Plain Mahalanobis for semantic segmentation:
          - Per-class means (μ_c) from ID training pixels
          - Shared covariance Σ (pooled across all pixels), with shrinkage
          - OOD score per image = mean over pixels of min_c Mahalanobis(x, μ_c)
            (higher = more OOD)
        """
        print("Initializing Mahalanobis OOD detector (plain)...")
        self.cfg = mmcv.Config.fromfile(config_file)
        self.cfg.model.pretrained = None
        self.model = init_segmentor(self.cfg, checkpoint_file, device=device)
        self.model.eval()

        self.device = device
        self.num_classes = 19
        self.eps = eps
        self.inference_times = []

        # Use a single deep backbone stage (stride 16) typical for ResNet C4
        # For MMSeg ResNet, backbone(x) returns a list [C1, C2, C3, C4]; index 2 = C3 (stride 16)
        self.backbone_stage_index = 2

        self.transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.class_means = None         # (num_classes, C)
        self.shared_precision = None    # (C, C)

    # ---------- I/O helpers ----------

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def _extract_backbone_stage(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns a feature map [1, C, H', W'] from the chosen backbone stage.
        NOTE: No L2 normalization here (plain Mahalanobis needs true covariance).
        """
        with torch.no_grad():
            backbone_outputs = self.model.backbone(img_tensor)
            if isinstance(backbone_outputs, (list, tuple)):
                feat = backbone_outputs[self.backbone_stage_index]
            else:
                feat = backbone_outputs
            # Downscale spatially to reduce memory/variance; keep enough resolution
            feat = F.adaptive_avg_pool2d(feat, (32, 64)).float()
        return feat  # [1, C, 32, 64]

    # ---------- Fitting (means & shared covariance) ----------

    def fit_training_data(self, training_paths: List[Tuple[str, str]]):
        """
        Compute per-class means μ_c and shared covariance Σ from ID training pixels.
        training_paths: list of (img_path, mask_path)
        """
        print(f"Fitting on {len(training_paths)} training images (plain Mahalanobis)...")

        class_sums = [None] * self.num_classes
        class_counts = [0] * self.num_classes
        all_feature_rows = []

        for i, (img_path, mask_path) in enumerate(training_paths):
            try:
                img_tensor = self.preprocess_image(img_path)
                feat = self._extract_backbone_stage(img_tensor)          # [1, C, H, W]
                _, C, H, W = feat.shape

                feat_flat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # [H*W, C]

                mask = Image.open(mask_path).resize((W, H), Image.NEAREST)
                mask_flat = np.array(mask, dtype=np.int64).reshape(-1)

                # Accumulate per-class sums/counts
                for cls in range(self.num_classes):
                    cls_idx = (mask_flat == cls)
                    if not np.any(cls_idx):
                        continue
                    f_cls = feat_flat[cls_idx]  # [N_cls, C]
                    if class_counts[cls] == 0:
                        class_sums[cls] = f_cls.sum(axis=0, dtype=np.float64)
                    else:
                        class_sums[cls] += f_cls.sum(axis=0, dtype=np.float64)
                    class_counts[cls] += f_cls.shape[0]

                # Collect for pooled covariance
                all_feature_rows.append(feat_flat)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(training_paths)} images")

            except Exception as e:
                print(f"  Warning: error processing {img_path}: {e}")

            # free memory
            del img_tensor
            torch.cuda.empty_cache()
            gc.collect()

        # Compute per-class means
        means = []
        for cls in range(self.num_classes):
            if class_counts[cls] > 0:
                means.append(class_sums[cls] / class_counts[cls])
            else:
                # class absent in training set; set mean to zeros
                # (pixels of this class won't help; detector still works for others)
                print(f"  Note: no pixels for class {cls}; setting mean to zeros.")
                means.append(np.zeros_like(class_sums[0]))
        self.class_means = np.stack(means, axis=0)  # [num_classes, C]

        # Shared covariance with shrinkage for stability
        X = np.concatenate(all_feature_rows, axis=0)  # [N_total, C]
        if X.shape[0] <= X.shape[1]:
            # Not enough samples; fall back to ridge
            print("  Warning: samples < dimensions; using ridge covariance.")
            cov = np.cov(X, rowvar=False) + 1e-2 * np.eye(X.shape[1])
            precision = np.linalg.pinv(cov)
        else:
            # Ledoit-Wolf shrinkage (recommended)
            lw = LedoitWolf().fit(X)
            precision = lw.precision_
        self.shared_precision = precision.astype(np.float64)  # [C, C]

        print("Fitting completed.")

    # ---------- Scoring (plain Mahalanobis) ----------

    def compute_image_anomaly_score(self, img_path: str) -> float:
        """
        Compute per-image OOD score:
          score = mean over pixels of min_c Mahalanobis(x, μ_c)
        Higher score = more OOD.
        """
        start_time = time.time()

        img_tensor = self.preprocess_image(img_path)
        feat = self._extract_backbone_stage(img_tensor)  # [1, C, H, W]
        _, C, H, W = feat.shape
        X = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # [N, C]
        MU = self.class_means  # [K, C]
        PREC = self.shared_precision  # [C, C]

        # Vectorized Mahalanobis to all class means: for each x, compute min_c (x-μ_c)^T Σ^{-1} (x-μ_c)
        # X: [N, C], MU: [K, C] -> compute distances in a batched way
        # We'll do it class-by-class to save memory
        min_qform = np.full(X.shape[0], np.inf, dtype=np.float64)
        for c in range(MU.shape[0]):
            diff = X - MU[c]                          # [N, C]
            q = np.einsum('ni,ij,nj->n', diff, PREC, diff, optimize=True)  # [N]
            np.minimum(min_qform, q, out=min_qform)

        # Per-pixel distances are positive; per-image score = mean distance (higher = more OOD)
        image_score = float(np.mean(min_qform))

        self.inference_times.append(time.time() - start_time)
        return image_score

    # ---------- Evaluation & reporting ----------

    def evaluate_ood_detection(self,
                               id_test_paths: List[Tuple[str, str]],
                               ood_paths: List[str],
                               training_paths: List[Tuple[str, str]]):
        """
        id_test_paths: list of (img_path, mask_path) for Cityscapes val
        ood_paths:    list of OOD image paths (no masks needed)
        training_paths: list of (img_path, mask_path) for Cityscapes train
        """
        print("Starting OOD evaluation (plain Mahalanobis)...")
        print(f"Training samples: {len(training_paths)}")
        print(f"ID test samples: {len(id_test_paths)}")
        print(f"OOD samples: {len(ood_paths)}")

        self.fit_training_data(training_paths)

        print("Scoring ID samples...")
        id_scores = []
        for i, (img_path, _) in enumerate(id_test_paths):
            s = self.compute_image_anomaly_score(img_path)
            id_scores.append(s)
            if (i + 1) % 10 == 0:
                print(f"  ID {i+1}/{len(id_test_paths)}")

        print("Scoring OOD samples...")
        ood_scores = []
        for i, img_path in enumerate(ood_paths):
            s = self.compute_image_anomaly_score(img_path)
            ood_scores.append(s)
            if (i + 1) % 10 == 0:
                print(f"  OOD {i+1}/{len(ood_paths)}")

        return self._calculate_metrics(np.array(id_scores), np.array(ood_scores))

    def _calculate_metrics(self, id_scores: np.ndarray, ood_scores: np.ndarray):
        """
        IMPORTANT: We define scores where HIGHER = MORE OOD.
        So with labels: ID=0, OOD=1, we pass scores directly to AUROC/AUPR.
        """
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)

        # FPR95: threshold that achieves 95% TPR on OOD (positives)
        fpr, tpr, thr = roc_curve(labels, scores)
        # find first index where TPR >= 0.95
        idx = np.argmax(tpr >= 0.95)
        fpr95 = float(fpr[idx]) if idx < len(fpr) else 1.0

        return {
            'auroc': float(auroc),
            'aupr': float(aupr),
            'fpr95': fpr95,
            'id_scores': id_scores.tolist(),
            'ood_scores': ood_scores.tolist(),
            'avg_inference_time': float(np.mean(self.inference_times)) if self.inference_times else 0.0,
            'timestamp': datetime.now().isoformat()
        }

    def save_results(self, results, output_path):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    def plot_results(self, results, output_path):
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        scores = np.concatenate([results['id_scores'], results['ood_scores']])
        labels = np.concatenate([np.zeros_like(results['id_scores']),
                                 np.ones_like(results['ood_scores'])])

        plt.figure(figsize=(15, 5))

        # --- ROC Curve ---
        plt.subplot(131)
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.plot(fpr, tpr, color="red", label=f"AUROC = {results['auroc']:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        # --- Precision-Recall Curve ---
        plt.subplot(132)
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, color="red", label=f"AUPR = {results['aupr']:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)

        # --- Score Distributions ---
        plt.subplot(133)
        plt.hist(results['id_scores'], bins=50, alpha=0.5, color="blue", label='ID', density=True)
        plt.hist(results['ood_scores'], bins=50, alpha=0.5, color="red", label='OOD', density=True)
        plt.xlabel('Mahalanobis Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to {output_path}")



# --------- Path utilities (robust for Cityscapes layout) ---------

def get_cityscapes_paths(img_root: str, mask_root: str) -> List[Tuple[str, str]]:
    """
    Recursively find matching image/mask pairs.
    Expects file names like *leftImg8bit.png and *gtFine_labelTrainIds.png
    """
    img_paths = sorted(glob.glob(os.path.join(img_root, "**", "*leftImg8bit.png"), recursive=True))
    pairs = []
    for img_path in img_paths:
        # Map image path to mask path by replacing directory and suffix
        base = os.path.basename(img_path)
        mask_name = base.replace("leftImg8bit", "gtFine_labelTrainIds")
        # replace parent dir name (LeftImg8bit or leftImg8bit -> gtFine_*)
        subdir = os.path.dirname(img_path).replace(img_root, "").lstrip(os.sep)
        # masks often mirror the city subdir structure; try same subpath
        mask_path = os.path.join(mask_root, subdir, mask_name)
        if not os.path.exists(mask_path):
            # fallback: search anywhere under mask_root
            candidates = glob.glob(os.path.join(mask_root, "**", mask_name), recursive=True)
            if len(candidates) > 0:
                mask_path = candidates[0]
            else:
                print(f"  Warning: mask not found for {img_path}")
                continue
        pairs.append((img_path, mask_path))
    return pairs


def get_ood_paths(ood_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(ood_dir, "**", e), recursive=True)
    return sorted(paths)


# ---------------- Main example ----------------

def main():
    config_file = "configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py"
    checkpoint_file = "checkpoints/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.pth"

    train_img_dir = "newdata/cityscape_newsubset/LeftImg8bit_train"
    train_mask_dir = "newdata/cityscape_newsubset/gtFine_train_cityscape"
    val_img_dir   = "newdata/cityscape_newsubset/leftImg8bit"
    val_mask_dir  = "newdata/cityscape_newsubset/gtFine_val_cityscape"
    ood_dir       = "newdata/lostandfound_newsubset/leftimg8bit"
    output_dir    = "ood_detection_project/results"

    os.makedirs(output_dir, exist_ok=True)

    detector = MahalanobisDetector(config_file, checkpoint_file)

    train_paths  = get_cityscapes_paths(train_img_dir, train_mask_dir)
    id_test_paths = get_cityscapes_paths(val_img_dir, val_mask_dir)
    ood_paths    = get_ood_paths(ood_dir)

    print(f"Found train pairs: {len(train_paths)}, ID val pairs: {len(id_test_paths)}, OOD: {len(ood_paths)}")

    results = detector.evaluate_ood_detection(
        id_test_paths=id_test_paths,
        ood_paths=ood_paths,
        training_paths=train_paths,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = os.path.join(output_dir, f"results_{timestamp}.json")
    results_png  = os.path.join(output_dir, f"plots_{timestamp}.png")

    detector.save_results(results, results_json)
    detector.plot_results(results, results_png)


if __name__ == "__main__":
    main()
