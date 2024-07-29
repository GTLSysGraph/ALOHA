#!/bin/bash

# 定义三个参数的值
model_names=('GraphMAE2')
datasets=('Cora_ml'  'Citeseer')
adaptive_attack_models=('svd_gcn' 'grand' 'gnn_guard' 'jaccard_gcn')

process_num=0
max_process_num=4

splits=(0 1 2 3 4)

Cora_ml_budgets_0=(646      760      760     754 )
Cora_ml_budgets_1=(684      757      760     641 )
Cora_ml_budgets_2=(646      721      760     717 )
Cora_ml_budgets_3=(608      746      760     755 )
Cora_ml_budgets_4=(532      715      760     760 )
Cora_ml_unit_ptb_0=(0.1274   0.1499   0.1499  0.1487)
Cora_ml_unit_ptb_1=(0.1349   0.1493   0.1499  0.1265)
Cora_ml_unit_ptb_2=(0.1274   0.1422   0.1499  0.1414)
Cora_ml_unit_ptb_3=(0.1199   0.1472   0.1499  0.1489)
Cora_ml_unit_ptb_4=(0.1050   0.1411   0.1499  0.1499)


Citeseer_budgets_0=(440       463      550     550 )
Citeseer_budgets_1=(357       521      550     550 )
Citeseer_budgets_2=(357       550      550     550 )
Citeseer_budgets_3=(495       550      550     519 )
Citeseer_budgets_4=(385       540      550     550 )
Citeseer_unit_ptb_0=(0.1200   0.1262   0.1499  0.1499)
Citeseer_unit_ptb_1=(0.0973   0.1420   0.1499  0.1499)
Citeseer_unit_ptb_2=(0.0973   0.1499   0.1499  0.1499)
Citeseer_unit_ptb_3=(0.1350   0.1499   0.1499  0.1415)
Citeseer_unit_ptb_4=(0.1050   0.1472   0.1499  0.1499)


# 遍历所有参数的组合
for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        for split in "${splits[@]}"; do
            # Dynamically reference the budget array using the split name
            budgets_array="${dataset}_budgets_${split}[@]"
            budgets=("${!budgets_array}")

            unit_ptb_array="${dataset}_unit_ptb_${split}[@]"
            unit_ptbs=("${!unit_ptb_array}")

            # for unit_ptb in "${unit_ptbs[@]}"; do
            #     echo "unit_ptb: $unit_ptb"
            # done
            # exit

            # # 判断变量长度是否相同 "${#budgets[@]}"表示参数长度  "${！budgets[@]}"为变量范围遍历 4个即 0 1 2 3
            # if [ "${#budgets[@]}" -ne "${#unit_ptbs[@]}" ]; then
            #     echo "Error: The number of datasets and a_values must be the same."
            #     exit 1
            # fi

            # 同时更新变量
            for i in "${!budgets[@]}"; do
                adaptive_attack_model="${adaptive_attack_models[$i]}"
                budget="${budgets[$i]}"
                unit_ptb="${unit_ptbs[$i]}"

                echo "运行顺序: $dataset $split $adaptive_attack_model $budget $unit_ptb"
                CUDA_VISIBLE_DEVICES=0 python Train_Runner.py   --task 'nodecls' \
                                                                --mode 'tranductive' \
                                                                --model_name               $model_name \
                                                                --dataset                  "Unit-${dataset}" \
                                                                --adaptive_attack_model    $adaptive_attack_model \
                                                                --split                    $split \
                                                                --scenario                 'poisoning' \
                                                                --budget                   $budget \
                                                                --unit_ptb                 $unit_ptb \
                                                                > "./logs/A800_Unit_Test/${model_name}/${dataset}_${split}_${adaptive_attack_model}_${budget}_${unit_ptb}.file"  &
                process_num=`expr $process_num + 1`
                process_num=`expr $process_num % $max_process_num`
                if [ $process_num == 0 ]
                then
                    wait
                fi 
            done
        done
    done
done