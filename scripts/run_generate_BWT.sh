#!/bin/bash


begin_id=0

for data_id in 3
do
    for ((ORDER=$begin_id; ORDER<14; ORDER++))
    do
        # 执行 Python 文件，传递参数 $i
        CUDA_VISIBLE_DEVICES=1 python generate_bwt_reasoning.py \
            --load_8bit \
            --base_model 'decapoda-research/llama-7b-hf' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            
    done
done