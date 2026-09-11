from __future__ import annotations

import argparse
from pathlib import Path
import xgboost as xgb

from .model_runner import add_common_args, run_regression_pipeline


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train XGBoost regressor for TyG_WHtR downstream task.")
	add_common_args(
		parser,
		default_model_name="xgboost",
		default_data_dir=Path("data/processed/downstream_tasks/TyG_WHtR"),
		default_results_dir=Path("data/results/TyG_WHtR"),
	)
	parser.add_argument(
		"--n-estimators",
		type=int,
		default=2000,
		help="Maximum number of boosting rounds.",
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=0.05,
		help="Boosting learning rate.",
	)
	parser.add_argument(
		"--max-depth",
		type=int,
		default=6,
		help="Maximum depth of each tree.",
	)
	parser.add_argument(
		"--early-stopping-rounds",
		type=int,
		default=50,
		help="Stop if validation score does not improve for this many rounds.",
	)
	parser.add_argument(
		"--verbose",
		type=int,
		default=100,
		help="Print evaluation metric every N rounds.",
	)
	return parser


def train_xgboost(
	args: argparse.Namespace,
	X_train,
	y_train,
	X_val,
	y_val,
) -> xgb.XGBRegressor:
	model = xgb.XGBRegressor(
		objective="reg:squarederror",
		n_estimators=args.n_estimators,
		learning_rate=args.learning_rate,
		max_depth=args.max_depth,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=42,
		eval_metric="rmse",
		early_stopping_rounds=args.early_stopping_rounds,
	)

	print("Training XGBoost model...")
	model.fit(
		X_train,
		y_train,
		eval_set=[(X_train, y_train), (X_val, y_val)],
		verbose=args.verbose,
	)
	return model


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	run_regression_pipeline(
		args,
		train_xgboost,
		feature_importances_fn=lambda model: model.feature_importances_,
	)


if __name__ == "__main__":
	main()
