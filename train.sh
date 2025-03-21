CUDA_VISIBLE_DEVICES=0
python Train_Runner.py \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'GraphMAE2' \
--dataset                  'Unit-Citeseer' \
--adaptive_attack_model    'jaccard_gcn' \
--split                    0 \
--scenario                 'poisoning' \
--budget                   550 \
--unit_ptb                 0.1499


# unit test
# --dataset                  'Unit-Cora_ml' \
# --adaptive_attack_model    'svd_gcn' \
# --split                    1 \
# --scenario                 'poisoning' \
# --budget                   684 \
# --unit_ptb                 0.1349


# common attack
# --dataset  'Attack-Cora' \
# --attack   'Meta_Self-0.25' \