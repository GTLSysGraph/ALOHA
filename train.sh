CUDA_VISIBLE_DEVICES=2
python Train_Runner.py \
--dataset  'Attack-Chameleon' \
--attack  'Meta_Self-0.0' \
--task 'node' \
--mode 'tranductive' \
--model_name 'GraphMAE'