
begin_id=0


for data_id in 1 2 3 4 5

do
    for ((ORDERR=$begin_id; ORDERR<15; ORDERR++))
    do
        
        WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --nproc_per_node=2 \
            --master_port=1234 \
            finetune_ContinualDST_T5XL.py \
            --num_epochs=5 \
            --micro_batch_size=8 \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDERR}
            

    done
done