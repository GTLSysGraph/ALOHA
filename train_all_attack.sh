MODEL_NAME=$1
DATASET_NAME=$2

CUDA_USE=(0 1)
 
# sh train_all_attack.sh MaskGAE Cora &

n=0
num=${#CUDA_USE[*]}

for i in 'Meta_Self' 'DICE' 'nettack' 'random'
do
    if [ $i == 'Meta_Self' ]
    then
        for ptb in 0.0 0.05 0.1 0.15 0.2 0.25
        do
            python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_"$i"_$2_$ptb.file" &
            wait
            n=`expr $n + 1`
            n=`expr $n % $num`
        done
        
        echo "All Metattack Train Done!"
        echo "###########################################################################################"
    elif [ $i == 'DICE' ]
    then
        for ptb in 0.0 0.1 0.2 0.3 0.4 0.5
        do
            python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_"$i"_$2_$ptb.file" &
            wait
            n=`expr $n + 1`
            n=`expr $n % $num`
        done
        echo "All DICE attack Train Done!"
        echo "###########################################################################################"
    elif [ $i == 'nettack' ]
    then
        for ptb in 0.0 1.0 2.0 3.0 4.0 5.0
        do
            python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_"$i"_$2_$ptb.file" &
            wait
            n=`expr $n + 1`
            n=`expr $n % $num`
        done
        echo "All nettack Train Done!"
        echo "###########################################################################################"        
    elif [ $i == 'random' ]
    then
        for ptb in 0.0 0.1 0.2 0.3 0.4 0.5
        do
            python Train_Runner.py --dataset "Attack-$2"  --attack  "$i-$ptb" --gpu_id ${CUDA_USE[$n]}  --model_name $1 >"./logs/$1/$1_"$i"_$2_$ptb.file" &
            wait
            n=`expr $n + 1`
            n=`expr $n % $num`
        done
        echo "All random Train Done!"
        echo "###########################################################################################" 
    fi
done
