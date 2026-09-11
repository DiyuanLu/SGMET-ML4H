#!/usr/bin/env bash
set -euo pipefail

conda run -n di-lab python -m src.nhanes.download_codebooks \
  --cycles 2011-2012 2013-2014 2015-2016 2017-2018 2019-2020 2021-2023
