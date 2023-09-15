CUDA_VISIBLE_DEVICES=2
python Train_Runner.py \
--dataset  'Attack-Cora' \
--attack  'Meta_Self-0.25' \
--task 'node' \
--mode 'tranductive' \
--model_name 'AutoRobustGAE'