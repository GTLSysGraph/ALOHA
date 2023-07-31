CUDA_VISIBLE_DEVICES=2
python Train_Runner.py \
--dataset  'Attack-Cora_ml' \
--attack  'Meta_Self-0.0' \
--task 'node' \
--mode 'tranductive' \
--model_name 'GraphMAE'