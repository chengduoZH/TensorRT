#!/bin/bash
set +xe
python ./helpers/convert_weights_pytorch.py -o ./bert
python ./helpers/generate_data.py -b 20 -s 10 -o ./20_10/
# python helpers/generate_tokens.py -t ./helpers/text.txt -o  ./
./build/sample_bert_model  -d ./ -d ./20_10/  --nheads  12