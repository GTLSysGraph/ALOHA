CUDA_VISIBLE_DEVICES=2
python Train_Runner.py \
--dataset  'Attack-Citeseer' \
--attack  'nettack-0.0' \
--task 'node' \
--mode 'tranductive' \
--model_name 'SPMGAE'