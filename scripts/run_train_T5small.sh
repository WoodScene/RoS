
begin_id=0


for data_id in 1 2 3 4 5

do
    for ((ORDERR=$begin_id; ORDERR<15; ORDERR++))
    do
        
        CUDA_VISIBLE_DEVICES=1 python finetune_continualDST_T5.py \
            --model_path '/your_model_path/t5base' \
            --num_epochs=5 \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            

    done
done