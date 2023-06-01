MODEL_NAME=$1

CUDA_USE=(0 1)
 
# sh train_all_graph.sh MaskGAE

n=0
process_num=0
num=${#CUDA_USE[*]}


# for loop in 1 2 3 4 5
# do
#     n=`expr $n + 1`
#     n=`expr $n % $num`
#     echo ${CUDA_USE[$n]}
# done

for i in 'IMDB-BINARY' 'IMDB-MULTI' 'PROTEINS' 'COLLAB' 'MUTAG' 'REDDIT-BINARY' 'NCI1'
do
    python Train_Runner.py --task 'graph'  --dataset $i  --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_graphcls_"$i".file" &
    process_num=`expr $process_num + 1`
    if [ $process_num == 4 ]
    then
        wait
    fi 
    n=`expr $n + 1`
    n=`expr $n % $num`
    sleep 5
done