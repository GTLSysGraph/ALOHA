MODEL_NAME=$1
DATASET_NAME=$2
TRAIN_MODE=$3

CUDA_USE=(0 2)
 

n=0
process_num=0
max_process_num=2


num=${#CUDA_USE[*]}

for i in 'DICE' 
do
    if [ $i == 'Meta_Self' ]
    then
        for ptb in 0.0 0.05 0.1 0.15 0.2 0.25
        do
            CUDA_VISIBLE_DEVICES=${CUDA_USE[$n]} python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --mode $3 --model_name $1 >"./logs/$1/$1_${i}_$2_${ptb}_$3.file" &
            process_num=`expr $process_num + 1`
            process_num=`expr $process_num % $max_process_num`
            if [ $process_num == 0 ]
            then
                wait
            fi 
            n=`expr $n + 1`
            n=`expr $n % $num`
            sleep 3
        done
        
        echo "All Metattack Train Done!"
        echo "###########################################################################################"
    elif [ $i == 'DICE' ]
    then
        for ptb in 0.0 0.1 0.2 0.3 0.4 0.5
        do
            CUDA_VISIBLE_DEVICES=${CUDA_USE[$n]} python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --mode $3 --model_name $1 >"./logs/$1/$1_${i}_$2_${ptb}_$3.file" &
            process_num=`expr $process_num + 1`
            process_num=`expr $process_num % $max_process_num`
            if [ $process_num == 0 ]
            then
                wait
            fi 
            n=`expr $n + 1`
            n=`expr $n % $num`
            sleep 3
        done

        echo "All DICE attack Train Done!"
        echo "###########################################################################################"
    elif [ $i == 'nettack' ]
    then
        for ptb in 0.0 1.0 2.0 3.0 4.0 5.0
        do
            CUDA_VISIBLE_DEVICES=${CUDA_USE[$n]} python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --mode $3 --model_name $1 >"./logs/$1/$1_${i}_$2_${ptb}_$3.file" &
            process_num=`expr $process_num + 1`
            process_num=`expr $process_num % $max_process_num`
            if [ $process_num == 0 ]
            then
                wait
            fi 
            n=`expr $n + 1`
            n=`expr $n % $num`
            sleep 3
        done

        echo "All nettack Train Done!"
        echo "###########################################################################################"        
    elif [ $i == 'random' ]
    then
        for ptb in 0.0 0.1 0.2 0.3 0.4 0.5
        do
            CUDA_VISIBLE_DEVICES=${CUDA_USE[$n]} python python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --mode $3 --model_name $1 >"./logs/$1/$1_${i}_$2_${ptb}_$3.file" &
            process_num=`expr $process_num + 1`
            process_num=`expr $process_num % $max_process_num`
            if [ $process_num == 0 ]
            then
                wait
            fi 
            n=`expr $n + 1`
            n=`expr $n % $num`
            sleep 3

        done
        echo "All random Train Done!"
        echo "###########################################################################################" 
    fi
done