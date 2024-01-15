CUDA_VISIBLE_DEVICES=0
python Train_Runner.py \
--dataset  'Attack-Citeseer' \
--attack  'Meta_Self-0.25' \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'GraphMAE2'
