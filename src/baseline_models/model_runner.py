from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from .baseline_model_evaluation import evaluate_classification, evaluate_regression


def add_common_args(
	parser: argparse.ArgumentParser,
	default_model_name: str,
	default_data_dir: Path | None = None,
	default_results_dir: Path | None = None,
) -> None:
	data_dir = default_data_dir or Path("data/processed/downstream_tasks/TyG_WHtR")
	results_dir = default_results_dir or Path("data/results/TyG_WHtR")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=data_dir,
		help="Directory containing X_train/X_val/X_test and y_*.parquet files.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default=default_model_name,
		help="Name of the model (used for column naming and output files).",
	)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=results_dir,
		help="Directory to save prediction artifacts.",
	)


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.Series]:
	x_path = data_dir / f"X_{split}.parquet"
	y_path = data_dir / f"y_{split}.parquet"

	X = pd.read_parquet(x_path)
	y = pd.read_parquet(y_path).squeeze()
	if isinstance(y, pd.DataFrame):
		y = y.iloc[:, 0]
	return X, y


def load_datasets(data_dir: Path) -> tuple[
	pd.DataFrame,
	pd.Series,
	pd.DataFrame,
	pd.Series,
	pd.DataFrame,
	pd.Series,
]:
	print("Loading data...")
	X_train, y_train = load_split(data_dir, "train")
	X_val, y_val = load_split(data_dir, "val")
	X_test, y_test = load_split(data_dir, "test")
	return X_train, y_train, X_val, y_val, X_test, y_test


def select_numeric_features(
	X_train: pd.DataFrame,
	X_val: pd.DataFrame,
	X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	X_train = X_train.select_dtypes(include=["number", "bool"])
	X_val = X_val.select_dtypes(include=["number", "bool"])
	X_test = X_test.select_dtypes(include=["number", "bool"])
	print(f"Features ready for training: {X_train.shape[1]} numeric columns.")
	return X_train, X_val, X_test


def save_tracked_predictions(
	y_true: pd.Series,
	y_pred: pd.Series,
	model_name: str,
	results_dir: Path,
) -> None:
	print("\nSaving tracked predictions...")
	results_dir.mkdir(parents=True, exist_ok=True)
	pred_df = pd.DataFrame(
		{
			"y_true": y_true.values,
			f"{model_name}_pred": y_pred,
		},
		index=y_true.index,
	)

	output_path = results_dir / f"{model_name}_predictions.parquet"
	pred_df.to_parquet(output_path)
	print(f"[OK] Predictions saved to {output_path}")


def save_tracked_predictions_classification(
	y_true: pd.Series,
	y_pred: pd.Series,
	y_proba: pd.DataFrame | None,
	model_name: str,
	results_dir: Path,
) -> None:
	print("\nSaving tracked predictions...")
	results_dir.mkdir(parents=True, exist_ok=True)
	pred_df = pd.DataFrame(
		{
			"y_true": y_true.values,
			f"{model_name}_pred": y_pred,
		},
		index=y_true.index,
	)

	if y_proba is not None:
		pred_df = pd.concat([pred_df, y_proba], axis=1)

	output_path = results_dir / f"{model_name}_predictions.parquet"
	pred_df.to_parquet(output_path)
	print(f"[OK] Predictions saved to {output_path}")


def run_regression_pipeline(
	args: argparse.Namespace,
	train_fn: Callable[[argparse.Namespace, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], object],
	feature_importances_fn: Callable[[object], object] | None = None,
) -> None:
	X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(args.data_dir)
	X_train, X_val, X_test = select_numeric_features(X_train, X_val, X_test)

	model = train_fn(args, X_train, y_train, X_val, y_val)
	y_pred = model.predict(X_test)

	feature_importances = None
	if feature_importances_fn is not None:
		feature_importances = feature_importances_fn(model)

	evaluate_regression(
		y_test,
		y_pred,
		feature_importances=feature_importances,
		feature_names=X_train.columns if feature_importances is not None else None,
	)

	save_tracked_predictions(y_test, y_pred, args.model_name, args.results_dir)


def run_classification_pipeline(
	args: argparse.Namespace,
	train_fn: Callable[[argparse.Namespace, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], object],
	feature_importances_fn: Callable[[object], object] | None = None,
) -> None:
	X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(args.data_dir)
	X_train, X_val, X_test = select_numeric_features(X_train, X_val, X_test)

	model = train_fn(args, X_train, y_train, X_val, y_val)
	if hasattr(model, "predict_proba"):
		y_proba = model.predict_proba(X_test)
		class_labels = list(model.classes_)
		y_pred = model.predict(X_test)
		proba_df = pd.DataFrame(
			y_proba,
			columns=[f"{args.model_name}_proba_{c}" for c in class_labels],
			index=y_test.index,
		)
	else:
		y_proba = None
		class_labels = None
		y_pred = model.predict(X_test)
		proba_df = None

	feature_importances = None
	if feature_importances_fn is not None:
		feature_importances = feature_importances_fn(model)

	evaluate_classification(
		y_test,
		y_pred,
		y_proba=y_proba,
		class_labels=class_labels,
		feature_importances=feature_importances,
		feature_names=X_train.columns if feature_importances is not None else None,
	)

	save_tracked_predictions_classification(
		y_test,
		y_pred,
		proba_df,
		args.model_name,
		args.results_dir,
	)
