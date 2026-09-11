#!/usr/bin/env python3
"""Clean-v5 FT-Transformer reference and parameter-matched controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/tokenized_nhanes_v5"
PILOT_OUTPUT = ROOT / "outputs/ft_transformer_v5_reference_and_matched"
FINAL_OUTPUT = ROOT / "outputs/ft_transformer_v5_fivefold_final_seed42"
OUTPUT = PILOT_OUTPUT
TARGET_TRAINABLE_PARAMS = 1_547_083
EXPECTED_FEATURES = 149
EXPECTED_TARGETS = 11


@dataclass(frozen=True)
class Architecture:
    name: str
    d_token: int
    n_layers: int
    n_heads: int
    d_ffn: int | None = None


ARCHITECTURES = {
    "reference": Architecture("reference", d_token=192, n_layers=3, n_heads=8),
    # Same depth and head count as the reference. Width and ReGLU hidden size
    # give 1,547,093 parameters, ten more than SGMET's 1,547,083.
    "matched": Architecture(
        "matched", d_token=232, n_layers=3, n_heads=8, d_ffn=327
    ),
    # Fold-0 validation-only architecture probes. Each changes one modelling
    # choice of interest; none is eligible for test scoring during selection.
    "matched_h4": Architecture(
        "matched_h4", d_token=232, n_layers=3, n_heads=4, d_ffn=327
    ),
    "depth4": Architecture("depth4", d_token=208, n_layers=4, n_heads=8),
    "compact": Architecture("compact", d_token=160, n_layers=3, n_heads=8),
}

FOLD0_PILOT_ORDER = ("matched", "matched_h4", "depth4", "compact")
FIVEFOLD_CONFIRMATION_ORDER = ("reference", "matched", "matched_h4")

PROTOCOL = {
    "seed": 42,
    "batch_size": 128,
    "epochs": 150,
    "patience": 15,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "weight_decay_policy": "exclude tokenizers, embeddings, biases, and normalization",
    "focal_gamma": 1.0,
    "selection": "validation_macro_auroc",
    "attention_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "activation": "ReGLU",
    "augmentation": "none; matched-missingness is a separate experiment",
    "natural_missingness": "categorical missing bucket plus learned continuous-missing embedding",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, first: bool, d_ffn: int | None):
        super().__init__()
        self.first = first
        self.attention_norm = nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(
            d_token,
            n_heads,
            dropout=PROTOCOL["attention_dropout"],
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_token)
        d_hidden = int(d_token * 4 / 3) if d_ffn is None else d_ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 2 * d_hidden),
            ReGLU(),
            nn.Dropout(PROTOCOL["ffn_dropout"]),
            nn.Linear(d_hidden, d_token),
        )
        self.residual_dropout = nn.Dropout(PROTOCOL["residual_dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = x if self.first else self.attention_norm(x)
        attention_output, _ = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )
        x = x + self.residual_dropout(attention_output)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(nn.Module):
    """Paper-default FT architecture with explicit but non-semantic missingness."""

    def __init__(
        self,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int],
        architecture: Architecture,
        n_targets: int,
    ):
        super().__init__()
        feature_type_ids = feature_type_ids.long()
        self.register_buffer("numeric_indices", torch.where(feature_type_ids == 0)[0])
        self.register_buffer("categorical_indices", torch.where(feature_type_ids == 1)[0])
        n_features = int(feature_type_ids.numel())
        n_numeric = int(self.numeric_indices.numel())
        d_token = architecture.d_token

        cat_cards = torch.tensor(
            [categorical_cardinalities[i] for i in self.categorical_indices.tolist()],
            dtype=torch.long,
        )
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), cat_cards.cumsum(0)[:-1]])
        self.register_buffer("category_offsets", offsets)
        self.register_buffer("category_cardinalities", cat_cards)

        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.feature_bias = nn.Parameter(torch.empty(n_features, d_token))
        self.numeric_missing_embedding = nn.Parameter(torch.empty(n_numeric, d_token))
        self.category_embeddings = nn.Embedding(int(cat_cards.sum()), d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token,
                    architecture.n_heads,
                    first=(i == 0),
                    d_ffn=architecture.d_ffn,
                )
                for i in range(architecture.n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_token)
        self.output_head = nn.Linear(d_token, n_targets)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (
            self.numeric_weight,
            self.feature_bias,
            self.numeric_missing_embedding,
            self.category_embeddings.weight,
            self.cls_token,
        ):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(
        self,
        numeric_values: torch.Tensor,
        categorical_codes: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_idx, cat_idx = self.numeric_indices, self.categorical_indices
        x_num = numeric_values[:, num_idx]
        m_num = missing_mask[:, num_idx].to(x_num.dtype)
        num_tokens = (
            x_num.unsqueeze(-1) * self.numeric_weight
            + self.feature_bias[num_idx]
            + m_num.unsqueeze(-1) * self.numeric_missing_embedding
        )

        cat_codes = categorical_codes[:, cat_idx].long()
        if torch.any(cat_codes < 0) or torch.any(cat_codes >= self.category_cardinalities):
            raise ValueError("categorical code outside tokenizer cardinality")
        cat_tokens = self.category_embeddings(cat_codes + self.category_offsets)
        cat_tokens = cat_tokens + self.feature_bias[cat_idx]

        tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_head(torch.relu(self.output_norm(x[:, 0])))


def metadata_for_fold(fold: int) -> dict:
    return torch.load(
        DATA / "cv_splits" / f"fold{fold}" / "tokenizer_metadata.pt",
        map_location="cpu",
        weights_only=False,
    )


def labels_to_tensor(targets: dict, target_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    columns = []
    for name in target_names:
        columns.append([float("nan") if v is None else float(v) for v in targets[name]])
    y = torch.tensor(np.asarray(columns, dtype=np.float32).T)
    mask = torch.isfinite(y)
    return torch.nan_to_num(y), mask


def load_split(fold: int, split: str, target_names: list[str]) -> TensorDataset:
    folder = DATA / "cv_splits" / f"fold{fold}"
    tokens = torch.load(folder / f"{split}_tokens.pt", map_location="cpu", weights_only=False)
    targets = torch.load(folder / f"{split}_targets.pt", map_location="cpu", weights_only=False)
    y, y_mask = labels_to_tensor(targets, target_names)
    return TensorDataset(
        tokens["numeric_values"].float(),
        tokens["categorical_codes"].long(),
        tokens["missing_mask"].bool(),
        y,
        y_mask,
    )


def model_for(architecture_name: str, metadata: dict) -> FTTransformer:
    return FTTransformer(
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        architecture=ARCHITECTURES[architecture_name],
        n_targets=len(metadata["target_columns"]),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_for(model: nn.Module) -> torch.optim.AdamW:
    tokenizer_prefixes = (
        "numeric_weight",
        "feature_bias",
        "numeric_missing_embedding",
        "category_embeddings.",
        "cls_token",
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exclude = parameter.ndim == 1 or name.startswith(tokenizer_prefixes)
        (no_decay if exclude else decay).append(parameter)
    assert decay and no_decay
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": PROTOCOL["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PROTOCOL["lr"],
    )


def experiment_config(architecture_name: str) -> dict:
    metadata = metadata_for_fold(0)
    model = model_for(architecture_name, metadata)
    count = parameter_count(model)
    return {
        "dataset": "tokenized_nhanes_v5",
        "physical_feature_count": EXPECTED_FEATURES,
        "target_count": EXPECTED_TARGETS,
        "architecture": asdict(ARCHITECTURES[architecture_name]),
        "protocol": PROTOCOL,
        "trainable_parameters": count,
        "sgmet_trainable_parameters": TARGET_TRAINABLE_PARAMS,
        "parameter_delta_fraction_vs_sgmet": (count - TARGET_TRAINABLE_PARAMS)
        / TARGET_TRAINABLE_PARAMS,
        "dataset_manifest_sha256": sha256(DATA / "MANIFEST.json"),
        "source_sha256": sha256(Path(__file__).resolve()),
        "test_policy": "not loaded during training; score once after all five validation runs",
    }


def architecture_dir(architecture_name: str) -> Path:
    return OUTPUT / architecture_name


def fold_dir(architecture_name: str, fold: int) -> Path:
    return architecture_dir(architecture_name) / f"fold{fold}"


def write_or_validate_config(architecture_name: str) -> None:
    destination = architecture_dir(architecture_name)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "CONFIG.json"
    expected = experiment_config(architecture_name)
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError(f"configuration guard failed for {path}")
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2))


def metrics_from_arrays(
    y: np.ndarray, mask: np.ndarray, probabilities: np.ndarray, target_names: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    aurocs, auprcs = [], []
    for j, name in enumerate(target_names):
        valid = mask[:, j].astype(bool)
        yj, pj = y[valid, j], probabilities[valid, j]
        short = name.removeprefix("label_")
        if valid.sum() and np.unique(yj).size == 2:
            auroc = float(roc_auc_score(yj, pj))
            auprc = float(average_precision_score(yj, pj))
        else:
            auroc = auprc = float("nan")
        result[f"{short}_auroc"] = auroc
        result[f"{short}_auprc"] = auprc
        aurocs.append(auroc)
        auprcs.append(auprc)
    result["macro_auroc"] = float(np.nanmean(aurocs))
    result["macro_auprc"] = float(np.nanmean(auprcs))
    return result


def focal_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for j in range(logits.shape[1]):
        valid = mask[:, j]
        if not valid.any():
            continue
        lj, yj = logits[valid, j], y[valid, j]
        bce = nn.functional.binary_cross_entropy_with_logits(lj, yj, reduction="none")
        p = torch.sigmoid(lj)
        pt = torch.where(yj > 0.5, p, 1.0 - p).clamp(1e-8, 1.0)
        losses.append(((1.0 - pt).pow(PROTOCOL["focal_gamma"]) * bce).mean())
    return torch.stack(losses).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_names: list[str],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    losses, ys, masks, probs = [], [], [], []
    for numeric, categorical, missing, y, y_mask in loader:
        numeric, categorical, missing = numeric.to(device), categorical.to(device), missing.to(device)
        y, y_mask = y.to(device), y_mask.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(numeric, categorical, missing)
            loss = focal_loss(logits, y, y_mask)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu())
        masks.append(y_mask.detach().cpu())
        probs.append(torch.sigmoid(logits.detach()).cpu())
    result = metrics_from_arrays(
        torch.cat(ys).numpy(),
        torch.cat(masks).numpy(),
        torch.cat(probs).numpy(),
        target_names,
    )
    result["loss"] = float(np.mean(losses))
    return result


def train_fold(architecture_name: str, fold: int) -> None:
    write_or_validate_config(architecture_name)
    destination = fold_dir(architecture_name, fold)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "DONE").exists():
        print(f"{architecture_name} fold {fold} already complete")
        return
    if any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {destination}")

    seed_all(PROTOCOL["seed"])
    metadata = metadata_for_fold(fold)
    target_names = list(metadata["target_columns"])
    train_data = load_split(fold, "train", target_names)
    val_data = load_split(fold, "val", target_names)
    generator = torch.Generator().manual_seed(PROTOCOL["seed"])
    train_loader = DataLoader(
        train_data,
        batch_size=PROTOCOL["batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=PROTOCOL["batch_size"], shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model_for(architecture_name, metadata).to(device)
    optimizer = optimizer_for(model)

    metrics_path = destination / "metrics.csv"
    best_auroc, best_record, bad_epochs = -math.inf, None, 0
    started = time.time()
    for epoch in range(PROTOCOL["epochs"]):
        train_metrics = run_epoch(model, train_loader, device, target_names, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, target_names, None)
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        improved = np.isfinite(val_metrics["macro_auroc"]) and (
            val_metrics["macro_auroc"] > best_auroc + 1e-6
        )
        if improved:
            best_auroc = val_metrics["macro_auroc"]
            best_record = row
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": architecture_name,
                    "model_state_dict": model.state_dict(),
                    "target_names": target_names,
                    "best_record": best_record,
                    "config": experiment_config(architecture_name),
                },
                destination / "best.pt",
            )
        else:
            bad_epochs += 1
        torch.save(
            {
                "epoch": epoch,
                "architecture": architecture_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "bad_epochs": bad_epochs,
                "best_record": best_record,
            },
            destination / "last.pt",
        )
        print(
            f"{architecture_name} fold={fold} epoch={epoch} "
            f"train={train_metrics['macro_auroc']:.4f} "
            f"val={val_metrics['macro_auroc']:.4f} "
            f"val_pr={val_metrics['macro_auprc']:.4f}",
            flush=True,
        )
        if bad_epochs >= PROTOCOL["patience"]:
            break

    if best_record is None:
        raise RuntimeError("no finite validation checkpoint was produced")
    done = {
        "architecture": architecture_name,
        "fold": fold,
        "selected_epoch": int(best_record["epoch"]),
        "epochs_run": epoch + 1,
        "selected_val_macro_auroc": best_record["val_macro_auroc"],
        "selected_val_macro_auprc": best_record["val_macro_auprc"],
        "runtime_minutes": (time.time() - started) / 60,
        "stopping_reason": "early_stopping" if bad_epochs >= PROTOCOL["patience"] else "epoch_cap",
        "test_scored": False,
        "trainable_parameters": parameter_count(model),
    }
    (destination / "DONE").write_text(json.dumps(done, indent=2))


def preflight() -> None:
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    assert manifest["physical_feature_count"] == EXPECTED_FEATURES
    reference_names = None
    for fold in range(5):
        metadata = metadata_for_fold(fold)
        names = list(metadata["feature_names"])
        assert len(names) == EXPECTED_FEATURES
        assert len(metadata["target_columns"]) == EXPECTED_TARGETS
        assert names == manifest["feature_order"]
        if reference_names is None:
            reference_names = names
        assert names == reference_names
        for split in ("train", "val", "test"):
            folder = DATA / "cv_splits" / f"fold{fold}"
            tokens = torch.load(folder / f"{split}_tokens.pt", map_location="cpu", weights_only=False)
            targets = torch.load(folder / f"{split}_targets.pt", map_location="cpu", weights_only=False)
            n = int(tokens["numeric_values"].shape[0])
            assert tokens["numeric_values"].shape == (n, EXPECTED_FEATURES)
            assert tokens["categorical_codes"].shape == (n, EXPECTED_FEATURES)
            assert tokens["missing_mask"].shape == (n, EXPECTED_FEATURES)
            assert all(len(targets[name]) == n for name in metadata["target_columns"])

    metadata = metadata_for_fold(0)
    sample = torch.load(
        DATA / "cv_splits/fold0/train_tokens.pt", map_location="cpu", weights_only=False
    )
    for name in ARCHITECTURES:
        model = model_for(name, metadata)
        count = parameter_count(model)
        logits = model(
            sample["numeric_values"][:4],
            sample["categorical_codes"][:4],
            sample["missing_mask"][:4],
        )
        assert logits.shape == (4, EXPECTED_TARGETS)
        assert torch.isfinite(logits).all()
        delta = (count - TARGET_TRAINABLE_PARAMS) / TARGET_TRAINABLE_PARAMS
        print(f"{name}: trainable_parameters={count:,}, delta_vs_sgmet={delta:+.2%}")
        if name in {"matched", "matched_h4"}:
            assert abs(count - TARGET_TRAINABLE_PARAMS) <= 10
    print("PREFLIGHT PASS: five folds, 149 features, 11 targets, finite forward passes")


def run_fold0_pilot() -> None:
    preflight()
    for architecture_name in FOLD0_PILOT_ORDER:
        train_fold(architecture_name, fold=0)


def run_fivefold_confirmation() -> None:
    global OUTPUT
    OUTPUT = FINAL_OUTPUT
    preflight()
    for architecture_name in FIVEFOLD_CONFIRMATION_ORDER:
        for fold in range(5):
            train_fold(architecture_name, fold)


def final_status() -> None:
    global OUTPUT
    OUTPUT = FINAL_OUTPUT
    status()


def status() -> None:
    for architecture_name in ARCHITECTURES:
        for fold in range(5):
            destination = fold_dir(architecture_name, fold)
            done = destination / "DONE"
            if done.exists():
                record = json.loads(done.read_text())
                print(
                    f"{architecture_name} fold={fold}: done, "
                    f"val AUROC={record['selected_val_macro_auroc']:.4f}"
                )
            elif (destination / "metrics.csv").exists():
                with (destination / "metrics.csv").open() as f:
                    rows = list(csv.DictReader(f))
                last = rows[-1]
                print(
                    f"{architecture_name} fold={fold}: incomplete, epoch={last['epoch']}, "
                    f"val AUROC={float(last['val_macro_auroc']):.4f}"
                )
            else:
                print(f"{architecture_name} fold={fold}: pending")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--final-status", action="store_true")
    parser.add_argument("--fold0-architecture-pilot", action="store_true")
    parser.add_argument("--fivefold-confirmation", action="store_true")
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.status:
        status()
    elif args.final_status:
        final_status()
    elif args.fold0_architecture_pilot:
        run_fold0_pilot()
    elif args.fivefold_confirmation:
        run_fivefold_confirmation()
    elif args.architecture is not None and args.fold is not None:
        train_fold(args.architecture, args.fold)
    else:
        parser.error("use --preflight, --status, or --architecture NAME --fold N")


if __name__ == "__main__":
    main()
