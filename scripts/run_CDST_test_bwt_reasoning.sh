#!/bin/bash

# 设置起始变量 从0开始，14截至，不是15截至了
begin_id=0

for data_id in 3
do
    # 循环从 begin_id 到 15
    #for ((ORDER=$begin_id; ORDER<14; ORDER++))
    for ORDER in 0 1 2 3 4 5 6 7 8 9 10 11 12 13
    do
        # 执行 Python 文件，传递参数 $i
        CUDA_VISIBLE_DEVICES=1 python generate_bwt_reasoning.py \
            --load_8bit \
            --base_model 'decapoda-research/llama-7b-hf' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            
    done
done