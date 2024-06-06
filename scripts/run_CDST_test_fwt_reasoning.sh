#!/bin/bash

# 设置起始变量 得从1开始
begin_id=1

for data_id in 3
do
    # 循环从 begin_id 到 15
    #for ((ORDER=$begin_id; ORDER<15; ORDER++))
    for ORDER in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    do
        # 执行 Python 文件，传递参数 $i
        CUDA_VISIBLE_DEVICES=1 python generate_fwt_reasoning.py \
            --load_8bit \
            --base_model 'decapoda-research/llama-7b-hf' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            
        
    done
done