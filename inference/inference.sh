#!/bin/bash
# Ultra Flash: Real-Time High-Resolution Streaming Video Generation
# One-click inference script

python inference.py \
    --config_path configs/self_forcing_dmd_4step.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/examples.txt \
    --output_folder outputs/ \
    --tiny_decoder \
    --torch_compile \
    --compile_sr_dit \
    --use_ema
