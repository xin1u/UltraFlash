#!/bin/bash
# Ultra Flash Long Video Inference (10s+ at 1K resolution)

python inference_sr.py \
    --config_path configs/longlive_inference_sr.yaml \
    --use_sparse_sr \
    --tiny_decoder \
    --torch_compile \
    --compile_sr_dit
