# 少样本音视频分类的语义调制提示学习

[English](README.md)

本仓库是论文 **Semantic Modulated Prompting for Few-Shot Audio-Visual
Classification** 的官方 PyTorch 实现。

## 方法简介

Semantic Modulated Prompting（SMP）面向少样本音视频分类中的过拟合、音视频
时序不同步和模态不平衡问题，包含两个部分：

- **P-AVeL**：在冻结的 ViT/CLIP 主干中加入参数高效的适配器，并使用文本语义
  提示引导音频和视频 token 的潜在注意力与融合。
- **P-PR**：利用文本原型动态调整音频和视频原型，再对较弱模态施加原型正则。

本公开版本只保留论文正式配置。在原测试代码中，它对应
`MODE=aux_fusion_layer` 和 `modulation_type=6`；其他消融和对比实验分支没有
包含在本仓库中。

## 安装

```bash
git clone https://github.com/DennisHgj/SMP_FSAVC.git
cd SMP_FSAVC
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

默认会下载 `vit_base_patch16_224.augreg_in21k` 和
`openai/clip-vit-large-patch14`。离线运行时，可以通过 `--vit-checkpoint`
指定本地 ViT 权重，通过 `--text-model` 指定本地 CLIP Hugging Face 目录。

## 数据格式

每个无表头 CSV 文件每行包含：

```text
视频片段ID,整数类别,语义提示文本
```

媒体文件路径为：

```text
<audio-dir>/<视频片段ID>.wav
<visual-dir>/<视频片段ID>/<帧图片>
```

源域目录需包含 `pretrain.csv` 和 `pretrain_test.csv`；目标少样本目录需包含
`fewshot.csv` 和 `fewshot_test.csv`。源域标签必须从 0 开始且连续。

## 源域预训练

```bash
python SourcePretraining.py \
  --dataset AVE \
  --num-classes 16 \
  --annotation-root /path/to/source_csv \
  --audio-dir /path/to/audio_files \
  --visual-dir /path/to/rgb_frames \
  --output-dir checkpoints
```

## 目标域少样本训练

以下命令执行论文中的 5-way 1-shot 协议。默认抽取 5 组类别，每组类别抽取
5 次支持集，共 25 次测试会话。

```bash
python TargetFS-AVC.py \
  --dataset AVE \
  --few-shot-root /path/to/target_csv \
  --audio-dir /path/to/audio_files \
  --visual-dir /path/to/rgb_frames \
  --pretrained-model checkpoints/smp_ave_source.pt \
  --n-way 5 \
  --k-shot 1
```

完整参数请运行 `python SourcePretraining.py --help` 或
`python TargetFS-AVC.py --help`。实验结果默认保存到 `results/`。

> **评估说明：** 为保持与论文测试代码一致，每次会话会报告少样本微调过程中
> 查询集的最佳准确率，因此查询标签参与了 epoch 选择。面向部署或严格的独立
> 测试时，请使用额外验证集选择 epoch，并且只在最后评估一次查询集。

## 测试

```bash
python -m unittest discover -s tests -v
```

引用信息、数据集划分、复现实验注意事项和许可证请参见[英文主页](README.md)。
