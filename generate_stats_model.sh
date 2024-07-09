dir=meta_models

mkdir -p $dir

log=$dir/train_stats.log

python run_experiment.py --dataset stats --generate_models --data_path datasets/stats/{}.csv --model_path $dir > $log
