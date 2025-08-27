#!/bin/sh

pip install -e .
pip install anemoi.datasets
python src/hirad/input_data/read_tp.py
