conda run -n di-lab python -m src.nhanes.prepare_features `
  --raw-root data/raw `
  --out-file data/processed/nhanes_2011_2023.parquet `
  --cycles 2011-2012 2013-2014 2015-2016 2017-2018 2019-2020 2021-2023

conda run -n di-lab python -m src.nhanes.build_downstream_labels `
  --in-file data/processed/nhanes_2011_2023.parquet `
  --out-file data/processed/nhanes_2011_2023.parquet
