CUDA_VISIBLE_DEVICES=1
python Train_Runner.py \
--dataset  'Cora' \
--attack  'Meta_Self-0.2' \
--task 'node' \
--mode 'tranductive' \
--model_name 'NASMGAE'