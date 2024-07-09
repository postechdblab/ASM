dir=AR_models

mkdir -p $dir

for table in $(ls datasets/stats/*csv); do
    y=${table%.*}
    z=${y##*/}
    log=$dir/train_stats_${z}.log
    python AR/run.py --run stats-single-${z} > $log
done
