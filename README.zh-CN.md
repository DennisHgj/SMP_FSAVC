# 少样本音视频分类的语义调制提示学习

[English](README.md) | [IEEE Xplore 论文](https://ieeexplore.ieee.org/abstract/document/11352954) | [DOI](https://doi.org/10.1109/TASLPRO.2026.3654246)

本仓库是论文 **Semantic Modulated Prompting for Few-Shot Audio-Visual
Classification** 的官方 PyTorch 实现。论文发表于 *IEEE Transactions on
Audio, Speech and Language Processing*，第 34 卷，第 723-736 页，2026 年。

## 方法简介

Semantic Modulated Prompting（SMP）面向少样本音视频分类中的过拟合、音视频
时序不同步和模态不平衡问题，包含两个部分：

- **P-AVeL**：在冻结的 ViT/CLIP 主干中加入参数高效的适配器，并使用文本语义
  提示引导音频和视频 token 的潜在注意力与融合。
- **P-PR**：利用文本原型动态调整音频和视频原型，再对较弱模态施加原型正则。

## 方法框架

![包含 P-AVeL 和 P-PR 的 SMP 总体框架](assets/smp_overview.png)

*P-AVeL 与 ViT 的 MLP 层并行，通过提示引导的潜在注意力实现三模态交互；
P-PR 使用语义原型调整音频和视频原型，并动态正则化较弱模态。*

## 主要实验结果

论文采用 5-way 少样本分类协议，并报告 25 次测试会话的平均准确率。使用 VLM
生成的语义提示时，SMP 在 AVE、VGGSound100 和 Kinetics-Sounds 的 1-shot
设置下分别达到 **85.74%**、**80.23%** 和 **74.37%**；对应的 20-shot 结果为
**96.46%**、**94.29%** 和 **85.92%**。SMP 在论文报告的全部设置中取得最佳
结果，同时仅包含 2.56M 个可训练参数。

![SMP 与现有方法在三个数据集上的性能对比](assets/main_results.png)

*论文 Table II 的主要对比结果。表中数值为平均分类准确率（%），上标为 25 次
测试会话的标准差。*

下图比较了 LAVISH（左）和 SMP（右）在随机 5-way 1-shot 任务上的 t-SNE
结果：上行为 VGGSound100，下行为 Kinetics-Sounds。SMP 得到了更清晰的类别
聚类和决策边界。

![LAVISH 与 SMP 的 t-SNE 可视化对比](assets/tsne_visualization.png)

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
`openai/clip-vit-large-patch14`。离线运行时，可通过 `--vit-checkpoint`
指定本地 ViT 权重，通过 `--text-model` 指定本地 CLIP Hugging Face 目录。

## 数据格式

每个无表头 CSV 文件至少包含三列，也可以包含可选的类别名称列：

```text
视频片段ID,整数类别,语义提示文本[,类别名称]
```

第三列会被原样送入文本编码器，既可以使用 [mPLUG-2](https://github.com/X-PLUG/mPLUG-2)
等 VLM 生成的 caption，也可以让所有样本都使用完全相同的静态提示
`a video of [label]`。其中 `[label]` 是不变的字面文本，不会替换成类别名称。
VGGSound100 样例中的第四列用于保留可读类别名称，不会送入 SMP 模型。

媒体文件路径为：

```text
<audio-dir>/<视频片段ID>.wav
<visual-dir>/<视频片段ID>/<帧图片>
```

源域目录需包含 `pretrain.csv` 和 `pretrain_test.csv`；目标少样本目录需包含
`fewshot.csv` 和 `fewshot_test.csv`。源域标签必须使用对应数据集从 0 开始的
标签空间；只有在缺失类别策略允许时，标签空间中才可以存在空类别。

论文使用的完整 CSV 划分和 mPLUG-2 captions 单独发布在
[SMP_FSAVC_Dataset 数据仓库](https://github.com/DennisHgj/SMP_FSAVC_Dataset)。
该仓库不重新发布原始媒体；请从上游数据集获取媒体并遵守其许可和使用条款。

当前 VGGSound100 划分保留 `0..59` 的 60 类源域标签空间，但标签 `14` 因无法
获取原始视频而没有样本。预训练默认给出一次警告，并保留该类的全零原型，以
复现原研究代码的行为。如需严格拒绝空类别，可使用
`--missing-class-policy error`。

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

## 引用

如果本项目对你的研究有帮助，请引用[正式发表的论文](https://ieeexplore.ieee.org/abstract/document/11352954)：

```bibtex
@ARTICLE{11352954,
  author={Huang, Guanjie and Cui, Yawen and Tsang, Danny H.K. and Wang, Wenwu and Liu, Li},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={Semantic Modulated Prompting for Few-Shot Audio-Visual Classification},
  year={2026},
  volume={34},
  pages={723-736},
  doi={10.1109/TASLPRO.2026.3654246}
}
```

## 许可证

本项目使用 [GNU General Public License v3.0](LICENSE)。
