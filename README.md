# Semantic Modulated Prompting for Few-Shot Audio-Visual Classification

This repository contains the official implementation of the paper **"Semantic Modulated Prompting for Few-Shot Audio-Visual Classification"**. 

> **📢 Notice / 说明**
> 
> The codebase is currently under reorganization. We are working on cleaning up and finalizing the code. **The latest version will be updated within this week.** Thank you for your interest and patience!
>
> 代码库目前正在整理中。我们正在清理代码并进行最终测试，**最新版本代码预计将在本周内更新完毕**。感谢您的关注与耐心等待！

## Introduction

This work proposes a novel Semantic Modulated Prompting (SMP) approach for few-shot audio-visual classification tasks.

## Datasets

The implementation supports the following datasets:
- **AVE** (Audio-Visual Event)
- **VGG-Sound**
- **Kinetics-Sound**

## Prerequisites

- Python 3.x
- PyTorch
- NumPy

## Usage

### Source Pretraining

Train the model on the source dataset:

```bash
python SourcePretraining.py --dataset AVE --num_classes 16
```

### Few-Shot Classification (Target)

Run few-shot adaptation/evaluation on the target dataset:

```bash
python TargetFS-AVC.py --dataset AVE
```

## Project Structure

- `dataloader/`: Contains dataset loading and preprocessing scripts.
- `models/`: Model architectures including the Semantic Modulated Prompting modules.
- `utils/`: Utility functions for training, testing, and logging.
- `SourcePretraining.py`: Script for the pre-training phase.
- `TargetFS-AVC.py`: Script for few-shot classification.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{SMP_FSAVC,
  title={Semantic Modulated Prompting for Few-Shot Audio-Visual Classification},
  author={Your Name and Authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
