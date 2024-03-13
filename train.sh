CUDA_VISIBLE_DEVICES=1
python Train_Runner.py \
--dataset  'Attack-Cora_ml' \
--attack  'DICE-0.5' \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'GraphMAE2'
