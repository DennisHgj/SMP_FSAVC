"""N-way K-shot target-set experiments for SMP."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import AudioVisualDataset
from models import SMPModel
from utils.common import (
    collate_audio_visual,
    resolve_device,
    save_json,
    seed_everything,
)
from utils.prototypes import estimate_prototypes
from utils.sampling import (
    read_annotations,
    sample_classes,
    sample_few_shot_task,
)
from utils.training import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate SMP on N-way K-shot FS-AVC tasks."
    )
    parser.add_argument(
        "--dataset", choices=("AVE", "VGGSound100", "Kinetics-Sounds"), default="AVE"
    )
    parser.add_argument("--few-shot-root", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", default="smp_5way_1shot")

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--class-rounds", type=int, default=5)
    parser.add_argument("--support-draws", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--begin-layer", type=int, default=4)
    parser.add_argument(
        "--adapter-location", choices=("attention", "mlp", "both"), default="mlp"
    )
    parser.add_argument("--attention-locations", default="before,after")
    parser.add_argument(
        "--tuning", choices=("none", "norm", "bias", "all"), default="bias"
    )
    parser.add_argument(
        "--vit-model", default="vit_base_patch16_224.augreg_in21k"
    )
    parser.add_argument("--vit-checkpoint", default=None)
    parser.add_argument("--text-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()
    if args.experiment_name == "smp_5way_1shot":
        args.experiment_name = f"smp_{args.n_way}way_{args.k_shot}shot"
    return args


def make_loader(
    annotations: pd.DataFrame,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = AudioVisualDataset(annotations, args.audio_dir, args.visual_dir)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_audio_visual,
    )


def load_source_weights(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    classifier_prefixes = (
        "fusion_classification_head.fc_action.",
        "fusion_classification_net.fc_action.",
    )
    state_dict = {
        name: value
        for name, value in state_dict.items()
        if not name.startswith(classifier_prefixes)
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_classifier_prefixes = (
        "fusion_classification_head.fc_action.",
        "fusion_classification_net.fc_action.",
    )
    missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_classifier_prefixes)
    ]
    unexpected = [
        name
        for name in incompatible.unexpected_keys
        if not name.startswith(allowed_classifier_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "The source checkpoint does not match this SMP configuration. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}. "
            "Check the adapter dimension, begin layer, adapter location, and "
            "latent-attention locations."
        )


def run_session(
    args: argparse.Namespace,
    support: pd.DataFrame,
    query: pd.DataFrame,
    device: torch.device,
) -> dict:
    model = SMPModel(
        num_classes=args.n_way,
        adapter_dim=args.adapter_dim,
        begin_layer=args.begin_layer,
        adapter_location=args.adapter_location,
        attention_locations=args.attention_locations,
        tuning=args.tuning,
        vit_model=args.vit_model,
        vit_checkpoint=args.vit_checkpoint,
        text_model=args.text_model,
    )
    load_source_weights(model, args.pretrained_model)
    model.to(device)

    train_loader = make_loader(support, args, shuffle=True)
    prototype_loader = make_loader(support, args, shuffle=False)
    query_loader = make_loader(query, args, shuffle=False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    prototypes = estimate_prototypes(model, prototype_loader, args.n_way, device)

    best_accuracy = float("-inf")
    best_epoch = 0
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
            args.n_way,
            device,
            previous=prototypes,
            momentum=args.prototype_momentum,
        )
        query_metrics = evaluate(model, query_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train": train_metrics, "query": query_metrics}
        )
        if query_metrics["accuracy"] > best_accuracy:
            best_accuracy = query_metrics["accuracy"]
            best_epoch = epoch

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "best_epoch": best_epoch,
        "best_query_accuracy": best_accuracy,
        "history": history,
    }


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if not args.pretrained_model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_model}")
    support_csv = args.few_shot_root / "fewshot.csv"
    query_csv = args.few_shot_root / "fewshot_test.csv"
    support_pool = read_annotations(support_csv)
    query_pool = read_annotations(query_csv)
    rng = np.random.default_rng(args.seed)

    session_results = []
    for class_round in range(args.class_rounds):
        selected_classes = sample_classes(support_pool, args.n_way, rng)
        for support_draw in range(args.support_draws):
            support, query, class_map = sample_few_shot_task(
                support_pool,
                query_pool,
                selected_classes,
                args.k_shot,
                rng,
            )
            print(
                f"Class round {class_round + 1}/{args.class_rounds}, "
                f"support draw {support_draw + 1}/{args.support_draws}: "
                f"classes={selected_classes}"
            )
            result = run_session(args, support, query, device)
            result.update(
                {
                    "class_round": class_round + 1,
                    "support_draw": support_draw + 1,
                    "selected_classes": selected_classes,
                    "class_map": class_map,
                }
            )
            session_results.append(result)
            print(f"  best query accuracy: {result['best_query_accuracy']:.2f}%")

    accuracies = np.asarray(
        [result["best_query_accuracy"] for result in session_results], dtype=float
    )
    summary = {
        "dataset": args.dataset,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "num_sessions": len(session_results),
        "mean_accuracy": float(accuracies.mean()),
        "standard_deviation": float(accuracies.std()),
        "minimum_accuracy": float(accuracies.min()),
        "maximum_accuracy": float(accuracies.max()),
        "sessions": session_results,
    }
    output_path = args.output_dir / f"{args.experiment_name}.json"
    save_json(output_path, summary)
    print(
        f"Mean accuracy: {summary['mean_accuracy']:.2f}% +/- "
        f"{summary['standard_deviation']:.2f} over {len(session_results)} sessions"
    )
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main(parse_args())
