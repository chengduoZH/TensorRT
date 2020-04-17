#!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np

parser.add_argument('-s', '--seqlen', type=int, required=True, help='The sequence length used to generate the tf record dataset')
parser.add_argument('-b', '--batchsize',type=int, required=True, help='The sequence length used to generate the tf record dataset')

opt = parser.parse_args()
slen = opt.seqlen
B = opt.batchsize
test_word_ids = np.random.randint(0, bert_config.vocab_size, (B, slen), dtype=np.int32)
test_input_mask = np.ones((B,slen), dtype=np.int32)
test_segment_ids = np.random.randint(0, bert_config.type_vocab_size, (B, slen), dtype=np.int32)

fd = {'input_ids:0' : test_word_ids,
        'input_mask:0':test_input_mask,
        'segment_ids:0':test_segment_ids}

with open(out_fn, 'wb') as output_file:
    count = 3
    output_file.write("{}\n".format(count).encode('ASCII'))
    idx = 0
    for k,v in fd.items():
        outname = '{}_{}'.format(k[:-2], idx)
        shape = v.shape
        shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
        output_file.write("{} 3 {} ".format(outname, shape_str).encode('ASCII'))
        output_file.write(v.tobytes())
        output_file.write("\n".encode('ASCII'))
