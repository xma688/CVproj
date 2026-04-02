import os, json
from pathlib import Path
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def run_vggt(image_dir, output_dir):
    image_dir = os.path.expanduser(image_dir)
    output_dir = os.path.expanduser(output_dir)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print("device:", device, "dtype:", dtype)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    print("images:", len(image_files))
    image_paths = [str(x) for x in image_files]

    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype):
        preds = model(images)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "images.npy"), images.cpu().numpy())
    np.save(os.path.join(output_dir, "intrinsics.npy"), preds["intrinsics"].cpu().numpy())
    np.save(os.path.join(output_dir, "extrinsics.npy"), preds["extrinsics"].cpu().numpy())
    if "world_points" in preds:
        np.save(os.path.join(output_dir, "world_points.npy"), preds["world_points"].cpu().numpy())

    with open(os.path.join(output_dir, "vggt_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(image_paths),
            "image_paths": image_paths,
            "device": device,
            "torch": torch.__version__
        }, f, indent=2, ensure_ascii=False)
    print("done:", output_dir)

if __name__ == "__main__":
    run_vggt(
        "~/my_storage_500G/CVproj/data/DL3DV-2/rgb",
        "~/my_storage_500G/CVproj/results/vggt_output"
    )