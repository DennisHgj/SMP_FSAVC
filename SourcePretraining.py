"""Source-set pretraining for Semantic Modulated Prompting."""

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import AudioVisualDataset
from models import SMPModel
from utils.common import (
    collate_audio_visual,
    resolve_device,
    save_json,
    seed_everything,
    trainable_parameter_count,
)
from utils.prototypes import estimate_prototypes
from utils.training import evaluate, train_one_epoch


DATASET_CLASS_COUNTS = {"AVE": 16, "VGGSound100": 60, "Kinetics-Sounds": 19}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain SMP on the source classes of an FS-AVC dataset."
    )
    parser.add_argument(
        "--dataset", choices=DATASET_CLASS_COUNTS, default="AVE"
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-name", default="smp_ave_source")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.0)
    parser.add_argument(
        "--missing-class-policy",
        choices=("allow", "error"),
        default="allow",
        help=(
            "How to handle source labels with no obtainable samples. "
            "'allow' keeps a zero prototype, matching the research code."
        ),
    )

    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--begin-layer", type=int, default=4)
    parser.add_argument(
        "--adapter-location", choices=("attention", "mlp", "both"), default="mlp"
    )
    parser.add_argument(
        "--attention-locations", default="before,after",
        help="Comma-separated P-AVeL latent-attention locations.",
    )
    parser.add_argument(
        "--tuning", choices=("none", "norm", "bias", "all"), default="bias"
    )
    parser.add_argument(
        "--vit-model", default="vit_base_patch16_224.augreg_in21k"
    )
    parser.add_argument("--vit-checkpoint", default=None)
    parser.add_argument("--text-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()
    if args.num_classes is None:
        args.num_classes = DATASET_CLASS_COUNTS[args.dataset]
    if args.experiment_name == "smp_ave_source" and args.dataset != "AVE":
        args.experiment_name = f"smp_{args.dataset.lower().replace('-', '_')}_source"
    return args


def make_data_loader(
    csv_path: Path,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = AudioVisualDataset(csv_path, args.audio_dir, args.visual_dir)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_audio_visual,
    )


def validate_source_label_space(
    train_csv: Path,
    validation_csv: Path,
    num_classes: int,
    missing_class_policy: str,
) -> list[int]:
    """Validate source labels before model loading or prototype extraction."""

    label_sets = {}
    for split_name, path in (
        ("training", train_csv),
        ("validation", validation_csv),
    ):
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] < 3:
            raise ValueError(
                f"{path} requires at least three columns: "
                "clip_id, label, semantic_prompt"
            )
        numeric = pd.to_numeric(frame.iloc[:, 1], errors="raise")
        if not numeric.eq(numeric.astype("int64")).all():
            raise ValueError(f"{path} contains non-integer class labels")
        labels = {int(value) for value in numeric}
        invalid = sorted(
            label for label in labels if not 0 <= label < num_classes
        )
        if invalid:
            raise ValueError(
                f"{split_name.capitalize()} labels must be in "
                f"[0, {num_classes - 1}]; found: {invalid}"
            )
        label_sets[split_name] = labels

    missing = sorted(set(range(num_classes)) - label_sets["training"])
    if missing:
        message = (
            f"Source training annotations contain no samples for classes {missing}."
        )
        if missing_class_policy == "error":
            raise ValueError(message)
        warnings.warn(
            message
            + " Their audio, visual, and text prototypes will remain zero to "
            "match the legacy research behavior.",
            RuntimeWarning,
            stacklevel=2,
        )

    validation_only = sorted(label_sets["validation"] - label_sets["training"])
    if validation_only:
        warnings.warn(
            "Validation annotations include classes absent from training: "
            f"{validation_only}",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    num_classes = args.num_classes
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    train_csv = args.annotation_root / "pretrain.csv"
    validation_csv = args.annotation_root / "pretrain_test.csv"
    for path in (train_csv, validation_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Required annotation file not found: {path}")
    missing_source_classes = validate_source_label_space(
        train_csv,
        validation_csv,
        num_classes,
        args.missing_class_policy,
    )

    train_loader = make_data_loader(train_csv, args, shuffle=True)
    prototype_loader = make_data_loader(train_csv, args, shuffle=False)
    validation_loader = make_data_loader(validation_csv, args, shuffle=False)

    model = SMPModel(
        num_classes=num_classes,
        adapter_dim=args.adapter_dim,
        begin_layer=args.begin_layer,
        adapter_location=args.adapter_location,
        attention_locations=args.attention_locations,
        tuning=args.tuning,
        vit_model=args.vit_model,
        vit_checkpoint=args.vit_checkpoint,
        text_model=args.text_model,
    ).to(device)
    print(f"Trainable parameters: {trainable_parameter_count(model):,}")

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.experiment_name}.pt"
    writer = SummaryWriter(log_dir=str(args.log_dir / args.experiment_name))

    prototypes = estimate_prototypes(
        model,
        prototype_loader,
        num_classes,
        device,
        allow_missing_classes=args.missing_class_policy == "allow",
    )
    best_accuracy = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            prototypes,
            device,
            args.alpha,
        )
        prototypes = estimate_prototypes(
            model,
            prototype_loader,
            num_classes,
            device,
            previous=prototypes,
            momentum=args.prototype_momentum,
            allow_missing_classes=args.missing_class_policy == "allow",
        )
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        for name, value in train_metrics.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in validation_metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)

        print(
            f"Epoch {epoch:03d} | train {train_metrics['accuracy']:.2f}% | "
            f"validation {validation_metrics['accuracy']:.2f}%"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                    "model_config": {
                        "adapter_dim": args.adapter_dim,
                        "begin_layer": args.begin_layer,
                        "adapter_location": args.adapter_location,
                        "attention_locations": args.attention_locations,
                        "tuning": args.tuning,
                        "vit_model": args.vit_model,
                        "text_model": args.text_model,
                    },
                    "source_label_config": {
                        "missing_class_policy": args.missing_class_policy,
                        "missing_source_classes": missing_source_classes,
                    },
                },
                checkpoint_path,
            )

    writer.close()
    save_json(
        args.output_dir / f"{args.experiment_name}.json",
        {
            "best_validation_accuracy": best_accuracy,
            "checkpoint": str(checkpoint_path),
            "missing_class_policy": args.missing_class_policy,
            "missing_source_classes": missing_source_classes,
            "history": history,
        },
    )
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main(parse_args())
