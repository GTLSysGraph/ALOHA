MODEL_NAME=$1

CUDA_USE=(0 1)
 
# sh train_all_graph.sh MaskGAE

n=0
process_num=0
max_process_num=4

num=${#CUDA_USE[*]}


# for loop in 1 2 3 4 5 6 7 8 9 10 11 12
# do
#     echo 1
#     process_num=`expr $process_num + 1`
#     process_num=`expr $process_num % $max_process_num`
#     if [ $process_num == 0 ]
#     then
#         echo "max! wait"
#     fi 
# done

for i in 'IMDB-BINARY' 'IMDB-MULTI' 'PROTEINS' 'COLLAB' 'MUTAG' 'REDDIT-BINARY' 'NCI1'
do
    python Train_Runner.py --task 'graph'  --dataset $i  --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_graphcls_"$i".file" &
    process_num=`expr $process_num + 1`
    process_num=`expr $process_num % $max_process_num`
    if [ $process_num == 0 ]
    then
        wait
    fi 
    n=`expr $n + 1`
    n=`expr $n % $num`
    sleep 5
done