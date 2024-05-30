# Continual Dialogue State Tracking via Reason-of-Select Distillation
Thank you for your interest in our work, and this is the original implementation of "Continual Dialogue State Tracking via Reason-of-Select Distillation", accepted to the ACL 2024 Findings.

## Local Setup
```
conda create -n CDST python=3.8
conda activate CDST
pip install -r requirements.txt
```

## Step 1. Teacher’s Reasoning Generation
The preprocessed SGD dataset is provided in the "/data" folder. You can then employ different teacher models to generate RoS reasoning.

* Get ChatGPT's rationales:
```ruby
./scripts/run_ChatGPT_reasoning.sh
```

* Get LLaMA-2-70B's rationales:
```ruby
./scripts/run_LLaMA2_70B_reasoning.sh
```

This step will generate $G$ ($G$ = 5 in our settings) candidate reasonings $\mathcal{R}_i$ as well as $N$ ($N$ = 6 in our settings) perturbed reasonings $\mathcal{PR}_i$. They will be saved in "./data" directory.



## Step 2.  Semantic Contrastive Reasoning Selection Stage
To ensure faithful teaching, we exploit semantic similarity to select optimal reasoning.


```ruby
./scripts/run_contrastive_selection.sh
```

Finally, obtained reasoning data is added to the original training dataset and the new reasoning dataset for model fine-tuning is constructed.

## Step 3. Training Student Models
We conducted experiments on four different student models:
### LLaMA-7B (`finetune_ContinualDST_LLaMA7B.py`)
```ruby
./scripts/run_train_LLaMA7B.sh
```
### FlanT5-XL (`finetune_ContinualDST_T5XL.py`)
```ruby
./scripts/run_train_FlanT5XL.sh
```
### T5-base (`finetune_ContinualDST_T5.py`)
```ruby
./scripts/run_train_T5base.sh
```
### T5-small (`finetune_ContinualDST_T5.py`)
```ruby
./scripts/run_train_T5small.sh
```

For LLaMA-7B and FlanT5-XL, we use [LoRA](https://github.com/microsoft/LoRA) to accelerate the speed of fine-tuning process. At the end of training, the student's fine-tuned weights will be stored in `$checkpoint_files`. We provide all the fine-tuning weights in the `Checkpoint_files` folder for reproducibility.

## Inference
We use three metrics to measure the performance of our model for Continual Learning. (You can directly load the weights that we have provided directly from the `\checkpoint` folder, and make inference.)

### **Avg.JGA** score
```ruby
./scripts/run_generate_avgJGA.sh
```
### Forward Transfer (**FWT**)
```ruby
./scripts/run_generate_FWT.sh
```
### Backward Transfer (**BWT**)
```ruby
./scripts/run_generate_BWT.sh
```
After inference, the generated prediction results will be stored at `\output` folder. 


## Evaluation
Then we can calculate these metrics by running
```ruby
./scripts/eval_avgJGA.sh
./scripts/eval_FWT.sh
./scripts/eval_BWT.sh
```


