

for data_id in 3
do
    CUDA_VISIBLE_DEVICES=0 python generate_average_reasoning.py \
        --load_8bit \
        --base_model 'decapoda-research/llama-7b-hf' \
        --dataset_id=${data_id}
done