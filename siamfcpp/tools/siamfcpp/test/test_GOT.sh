#!/usr/bin/env bash

python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_shufflenetv2x0_5-got.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_shufflenetv2x1_0-got.yaml'
