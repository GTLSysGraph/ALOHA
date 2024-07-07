CUDA_VISIBLE_DEVICES=0
python Train_Runner.py \
--dataset  'Attack-Cora' \
--attack   'DICE-0.0' \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'GraphMAE'

# unit test
# --dataset                 'Unit-Cora_ml' \
# --adaptive_attack_model   'clean' \
# --split                    4 \
# --scenario                'poisoning' \

# common attack
# --dataset  'Attack-Citeseer' \
# --attack   'Meta_Self-0.25' \