# 3201Project

## Part 1 Plan A 操作说明

请先确认当前环境中已经正确安装 `colmap`：

```bash
conda install conda-forge::colmap
```

### 1. 当前支持的数据集

脚本目前支持以下数据集 key：

- `re10k` → `data/Re10k-1/images`
- `dl3dv` → `data/DL3DV-2/rgb`
- `waymo_front` → `data/405841/FRONT/rgb`

### 2. 运行 COLMAP

Re10k-1
```bash
bash scripts/run_colmap.sh re10k
```

DL3DV-2
```bash
bash scripts/run_colmap.sh dl3dv
```

Waymo FRONT
```bash
bash scripts/run_colmap.sh waymo_front
```

脚本支持以下可选参数：

```bash
bash scripts/run_colmap.sh [dataset_key] [matcher] [use_gpu]
```

参数说明：

- `dataset_key`：`re10k | dl3dv | waymo_front`
- `matcher`：`sequential | exhaustive`
- `use_gpu`：`1 | 0`

### 3. 检查所有 sparse reconstruction

为了检查所有生成的 sparse 模型，并自动选出最优模型，可以运行：

```bash
bash scripts/inspect_colmap.sh re10k 0
bash scripts/inspect_colmap.sh dl3dv 0
bash scripts/inspect_colmap.sh waymo_front 0
```

该脚本会对每个 `sparse/N` 模型统计：

- 注册图像数量（Registered images）
- 稀疏点云数量（Sparse points3D）

并按照以下规则自动选择最佳模型：

- 优先选择注册图像数最多的模型
- 如果注册图像数相同，则选择稀疏点数更多的模型

### 4. 保存最佳模型的 TXT 格式结果

如果希望把最佳模型额外导出为 TXT 格式，便于检查内容，可以运行：

```bash
bash scripts/inspect_colmap.sh re10k 1
bash scripts/inspect_colmap.sh dl3dv 1
bash scripts/inspect_colmap.sh waymo_front 1
```

这会生成：

```text
outputs/colmap/<SCENE_NAME>/best_sparse_txt/
```

## 3DGS 数据整理与训练准备

完成了 Plan A 从 COLMAP 到 3DGS 的数据对接，主要包括：

- 新增 `scripts/organize_3dgs_scene.sh`：将三个数据集统一整理为 3DGS 所需结构：

```text
scenes_3dgs/<SCENE>/
├── images
└── sparse/0
```

- 检查 COLMAP 相机模型后发现初始结果为 `SIMPLE_RADIAL`，因此不能直接用于官方 3DGS 训练。
- 新增 `scripts/convert_on_scenes_3dgs.sh`：在 `scenes_3dgs/<SCENE>` 上直接执行 `convert.py --skip_matching`，完成 undistort 和相机模型转换。
- 三个数据集现已全部成功转换为 `PINHOLE` 格式，可直接用于 3DGS。
- 新增 `scripts/check_3dgs_scene.sh`：用于检查目录结构、导出 `cameras.txt` 并确认相机模型是否兼容。

### 5. 整理为 3DGS 输入结构

```bash
bash scripts/organize_3dgs_scene.sh Re10k-1 0
bash scripts/organize_3dgs_scene.sh DL3DV-2 0
bash scripts/organize_3dgs_scene.sh 405841_FRONT 0
```

其中第二个参数表示使用的 COLMAP sparse model id，例如 `0` 或 `1`。

### 6. 执行 convert

```bash
bash scripts/convert_on_scenes_3dgs.sh Re10k-1
bash scripts/convert_on_scenes_3dgs.sh DL3DV-2
bash scripts/convert_on_scenes_3dgs.sh 405841_FRONT
```

执行后会在 `scenes_3dgs/<SCENE>` 下保留：

- `input/`：原始输入图像
- `distorted/`：convert 前的 COLMAP 数据
- `images/`：undistort 后供 3DGS 训练使用的图像
- `sparse/0/`：转换后的 COLMAP 模型

### 7. 检查是否已转为 `PINHOLE`

```bash
bash scripts/check_3dgs_scene.sh Re10k-1
bash scripts/check_3dgs_scene.sh DL3DV-2
bash scripts/check_3dgs_scene.sh 405841_FRONT
```

如果输出中出现：

```text
Detected camera model : PINHOLE
```

则说明该场景已经可以直接用于官方 3DGS 训练。

## 3DGS 训练与评测脚本

新增两个脚本：

- `scripts/train_3dgs.sh`
- `scripts/eval_3dgs.sh`

这两个脚本都通过参数控制数据集与实验方案，适合后续对比 Plan A / Plan B。

### 8. 训练脚本 `train_3dgs.sh`

用法：

```bash
bash scripts/train_3dgs.sh [SCENE] [PLAN]
```

支持的 `SCENE`：

- `DL3DV-2`
- `Re10k-1`
- `405841_FRONT`

支持的 `PLAN`：

- `PlanA`
- `PlanB`

示例：

```bash
bash scripts/train_3dgs.sh DL3DV-2 PlanA
bash scripts/train_3dgs.sh Re10k-1 PlanA
bash scripts/train_3dgs.sh 405841_FRONT PlanA
```

训练脚本会自动：

- 读取 `scenes_3dgs/<SCENE>` 作为 source path
- 将输出保存到 `outputs/3dgs/<PLAN>/<SCENE>`
- 开启 `--eval`，使用 3DGS 的 train/test split
- 记录训练日志、开始结束时间与总耗时
- 在多个迭代点保存中间模型，便于比较收敛速度
- 在多个迭代点记录 test-set 的评估信息

输出目录示例：

```text
outputs/3dgs/PlanA/DL3DV-2/
└── logs/
    ├── train_console.log
    └── train_meta.txt
```

### 9. 评测脚本 `eval_3dgs.sh`

用法：

```bash
bash scripts/eval_3dgs.sh [SCENE] [PLAN]
```

示例：

```bash
bash scripts/eval_3dgs.sh DL3DV-2 PlanA
bash scripts/eval_3dgs.sh Re10k-1 PlanA
bash scripts/eval_3dgs.sh 405841_FRONT PlanA
```

评测脚本会自动：

- 调用 `render.py`
- 使用 `--skip_train`，只渲染测试集
- 调用 `metrics.py`，在测试集上计算指标
- 保存渲染日志与评测日志

输出目录示例：

```text
outputs/3dgs/PlanA/DL3DV-2/
└── logs/
    ├── render_test_console.log
    ├── metrics_test_console.log
    └── eval_meta.txt
```

### 10. 推荐运行顺序

建议在项目根目录下执行脚本：

```bash
cd /path/to/project_root
```
项目根目录应有如下结构：
```bash
project_root/
├── data
├── gaussian-splatting
├── outputs
├── scenes_3dgs
└── scripts
```
然后按以下顺序运行：

```bash
# 1) 整理 3DGS 输入结构
bash scripts/organize_3dgs_scene.sh DL3DV-2 0

# 2) 执行 convert
bash scripts/convert_on_scenes_3dgs.sh DL3DV-2

# 3) 检查相机模型
bash scripts/check_3dgs_scene.sh DL3DV-2

# 4) 训练
bash scripts/train_3dgs.sh DL3DV-2 PlanA

# 5) 评测
bash scripts/eval_3dgs.sh DL3DV-2 PlanA
```
