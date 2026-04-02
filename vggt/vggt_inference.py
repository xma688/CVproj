import os
import json
from pathlib import Path
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def run_vggt(image_dir, output_dir, chunk_size=4):
    """
    分块运行 VGGT 推理，避免显存溢出。

    Args:
        image_dir: 图像文件夹路径（支持 ~ 展开）
        output_dir: 输出文件夹路径（支持 ~ 展开）
        chunk_size: 每次处理的图像数量，显存不足时减小该值（例如 2 或 1）
    """
    # 展开路径中的 ~
    image_dir = os.path.expanduser(image_dir)
    output_dir = os.path.expanduser(output_dir)

    # 设备与精度设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print("device:", device, "dtype:", dtype)

    # 加载模型（不传递 frames_chunk_size，因为该参数不存在）
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # 获取所有图像路径（支持 jpg 和 png）
    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"Total images found: {len(image_files)}")

    # 准备存储所有块的结果
    all_intrinsics = []
    all_extrinsics = []
    all_world_points = []

    # 分块处理
    for start_idx in range(0, len(image_files), chunk_size):
        end_idx = min(start_idx + chunk_size, len(image_files))
        chunk_paths = image_files[start_idx:end_idx]
        print(f"Processing chunk {start_idx//chunk_size + 1}/{(len(image_files)-1)//chunk_size + 1} "
              f"(images {start_idx+1} to {end_idx})")

        # 加载并预处理当前块图像
        # load_and_preprocess_images 返回形状 (N, 3, H, W)
        images = load_and_preprocess_images([str(p) for p in chunk_paths]).to(device)
        # 增加 batch 维度：模型期望 (1, N, 3, H, W)
        images = images.unsqueeze(0)

        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype):
            preds = model(images)

        # 收集结果（移到 CPU 并转换为 numpy）
        all_intrinsics.append(preds["intrinsics"].cpu().numpy())      # 形状 (1, N, ...)
        all_extrinsics.append(preds["extrinsics"].cpu().numpy())      # 形状 (1, N, ...)
        if "world_points" in preds:
            all_world_points.append(preds["world_points"].cpu().numpy())

        # 可选：手动清理显存（避免碎片）
        del images, preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 合并所有块的结果（沿序列维度拼接，即 axis=1）
    final_intrinsics = np.concatenate(all_intrinsics, axis=1)
    final_extrinsics = np.concatenate(all_extrinsics, axis=1)
    final_world_points = np.concatenate(all_world_points, axis=1) if all_world_points else None

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "intrinsics.npy"), final_intrinsics)
    np.save(os.path.join(output_dir, "extrinsics.npy"), final_extrinsics)
    if final_world_points is not None:
        np.save(os.path.join(output_dir, "world_points.npy"), final_world_points)

    # 保存元数据（不再保存 images.npy 以节省磁盘空间）
    with open(os.path.join(output_dir, "vggt_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(image_files),
            "image_paths": [str(p) for p in image_files],
            "chunk_size": chunk_size,
            "device": device,
            "torch": torch.__version__
        }, f, indent=2, ensure_ascii=False)

    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    run_vggt(
        "~/my_storage_500G/CVproj/data/DL3DV-2/rgb",
        "~/my_storage_500G/CVproj/results/vggt_output",
        chunk_size=4   # 根据显存调整：2、1 等
    )