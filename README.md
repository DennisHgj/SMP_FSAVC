# Semantic Modulated Prompting for Few-Shot Audio-Visual Classification

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)

Official PyTorch implementation of **Semantic Modulated Prompting (SMP)** for
few-shot audio-visual classification (FS-AVC).

[中文说明](README.zh-CN.md)

## Overview

SMP addresses three difficulties in FS-AVC: overfitting, audio-visual temporal
asynchrony, and modality imbalance. It has two trainable components on top of
mostly frozen ViT and CLIP encoders:

- **Prompt-refined Audio-Visual efficient Learner (P-AVeL):** adapter blocks
  use semantic text prompts to guide latent audio-visual alignment and fusion.
- **Prompt-tuned Prototypical Regularization (P-PR):** text prototypes tune
  audio and visual prototypes before dynamically regularizing the weaker
  modality.

This release contains only the paper configuration. In the original research
workspace, that configuration was selected by `MODE=aux_fusion_layer` and
`modulation_type=6`; those experiment switches and unused ablation branches
have intentionally been removed here.

## Repository layout

```text
.
|-- SourcePretraining.py       # source-set pretraining
|-- TargetFS-AVC.py            # N-way K-shot target-set experiments
|-- dataloader/
|   `-- dataset.py             # audio, frame, and caption loading
|-- models/
|   |-- smp.py                 # complete SMP model
|   |-- p_avel.py              # P-AVeL adapter
|   |-- prompt_attention.py    # prompt-guided latent attention
|   |-- smp_fusion_block.py    # tri-stream Transformer block
|   `-- classification_head.py
|-- utils/
|   |-- common.py              # reproducibility and batching helpers
|   |-- prototypes.py          # prototype estimation and P-PR
|   |-- sampling.py            # reproducible few-shot task sampling
|   `-- training.py            # train/evaluation loops
`-- tests/                     # lightweight unit tests
```

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/DennisHgj/SMP_FSAVC.git
cd SMP_FSAVC
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

By default, `timm` downloads `vit_base_patch16_224.augreg_in21k` and
Transformers downloads `openai/clip-vit-large-patch14`. For an offline run,
pass a local ViT checkpoint with `--vit-checkpoint` and a local Hugging Face
CLIP directory with `--text-model`.

## Data preparation

The code expects 16 kHz-compatible WAV files, extracted RGB frames, and
headerless CSV annotation files. Each CSV row has three fields:

```text
clip_id,integer_label,semantic_prompt
```

The corresponding media are resolved as:

```text
<audio-dir>/<clip_id>.wav
<visual-dir>/<clip_id>/<frame files>
```

Frames are sorted by filename and eight uniformly spaced frames are used by
default. Source-set labels must be contiguous and zero-based. A typical
annotation layout is:

```text
annotations/
|-- source/
|   |-- pretrain.csv
|   `-- pretrain_test.csv
`-- target/
    |-- fewshot.csv
    `-- fewshot_test.csv
```

The paper uses the following source/target split sizes:

| Dataset | Source classes | Target classes |
| --- | ---: | ---: |
| AVE | 16 | 12 |
| VGGSound100 | 60 | 40 |
| Kinetics-Sounds | 19 | 13 |

Dataset downloads and captions are not redistributed by this repository.
Please follow each dataset's license and usage terms. Semantic prompts may be
VLM-generated captions or constant prompts such as `A video of <label>`.

## Source-set pretraining

The defaults reproduce the main architecture settings from the paper:
P-AVeL starts at Transformer block 4, uses a downsampling dimension of 16,
applies prompt-guided latent attention before and after modality convolutions,
and places P-AVeL parallel to the MLP sublayer.

```bash
python SourcePretraining.py \
  --dataset AVE \
  --num-classes 16 \
  --annotation-root /path/to/annotations/source \
  --audio-dir /path/to/audio_files \
  --visual-dir /path/to/rgb_frames \
  --output-dir checkpoints
```

Use `python SourcePretraining.py --help` for all options. The best validation
checkpoint is written to `<output-dir>/<experiment-name>.pt`.

## Few-shot target training

The command below runs the paper's 5-way 1-shot protocol. Five class draws and
five support-set draws per class set produce 25 sessions.

```bash
python TargetFS-AVC.py \
  --dataset AVE \
  --few-shot-root /path/to/annotations/target \
  --audio-dir /path/to/audio_files \
  --visual-dir /path/to/rgb_frames \
  --pretrained-model checkpoints/smp_ave_source.pt \
  --n-way 5 \
  --k-shot 1
```

Session metrics and aggregate mean/standard deviation are saved as JSON in
`results/` by default. To reproduce 5-, 10-, or 20-shot experiments, change
only `--k-shot`.

> **Evaluation note:** the released research protocol selects the best query
> accuracy across fine-tuning epochs for each session. This matches the paper's
> reported experimental code, but it uses query labels for epoch selection.
> For deployment or a strict held-out evaluation, select the epoch using a
> separate validation split and evaluate the query set only once.

## Reproducibility notes

- The default seed is 42, as in the paper.
- The default optimizer is Adam with learning rate `1e-4` and batch size 16.
- ViT weights are frozen except the audio projection and bias-tuning
  parameters; the CLIP text encoder remains frozen. P-AVeL and the classifier
  are trainable.
- A source checkpoint can be either the checkpoint dictionary produced by
  this release or the raw `aux_fusion_layer` state dictionary used by the
  cleaned research scripts. Other ablation checkpoints are not compatible.
- Exact results also depend on media preprocessing, generated captions,
  package versions, and hardware.

## Tests

```bash
python -m unittest discover -s tests -v
```

These tests cover prototype computation, P-PR loss construction, deterministic
few-shot sampling, and latent-attention tensor shapes. Full training requires
the datasets and pretrained encoder weights.

## Citation

If this work is useful in your research, please cite the manuscript. Update
the venue fields when the final bibliographic record becomes available.

```bibtex
@article{huang2026semantic,
  title   = {Semantic Modulated Prompting for Few-Shot Audio-Visual Classification},
  author  = {Huang, Guanjie and Cui, Yawen and Tsang, Danny Hin Kwok and Wang, Wenwu and Liu, Li},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
