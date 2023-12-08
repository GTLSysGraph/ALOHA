CUDA_VISIBLE_DEVICES=0
python Train_Runner.py \
--dataset  'Attack-ogbn-arxiv' \
--attack  'DICE-0.5' \
--task 'nodecls' \
--mode 'mini_batch' \
--model_name 'GraphMAE2'
