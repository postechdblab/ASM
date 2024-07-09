dir=AR_models

mkdir -p $dir

gpu=0

for table in $(ls ./datasets/stack); do
    y=${table%.*}
    z=${y##*/}
    if [ "$z" != "so_pg13" ] && [ "$z" != "stack_index" ]; then 
        echo $z
        log=$dir/train_stack_${z}.log
        CUDA_VISIBLE_DEVICES=$gpu python AR/run.py --run stack-single-${z} > $log &
        gpu=$(($gpu+1))
        gpu=$(($gpu%8))
    fi
done

wait