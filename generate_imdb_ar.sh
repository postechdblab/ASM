dir=AR_models

mkdir -p $dir

gpu=0

for table in $(ls ./datasets/imdb/*csv); do
    y=${table%.*}
    z=${y##*/}
    echo $z
    log=$dir/train_imdb_${z}.log
    CUDA_VISIBLE_DEVICES=$gpu python AR/run.py --run imdb-single-${z} > $log &
    gpu=$(($gpu+1))
    gpu=$(($gpu%8))
done

wait
