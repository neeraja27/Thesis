from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
import torch
import os
from datetime import datetime

# Paths to config and checkpoint
config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.pth'

# Verify GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Ensure the container has GPU access via 'hare run --gpus all'.")

# Initialize model on GPU
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
model.eval()  # Set model to evaluation mode

# Test image from subsetdata/cityscapes_subset/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
test_img = 'data/cityscapes_subset/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
if not os.path.exists(test_img):
    print(f"Test image {test_img} not found. Please check the path in data/cityscapes_subset/leftImg8bit/val/frankfurt/")
else:
    # Run inference
    result = inference_segmentor(model, test_img)
    print("Inference completed successfully!")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'segmentation_result_{timestamp}.png')
    model.show_result(test_img, result, out_file=out_file)
    print(f"Segmentation output saved to {out_file}")