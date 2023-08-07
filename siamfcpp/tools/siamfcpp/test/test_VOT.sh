#!/usr/bin/env bash
python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_tinyconv.yaml'

# python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet-multi_temp.yaml'
# python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_googlenet-new.yaml'
# python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_googlenet_bn.yaml'
# python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_googlenet-multi_temp.yaml'
# python3 ./main/test.py --config 'experiments/siamfcpp/test/vot/siamfcpp_tinyconv-multi_temp.yaml'
