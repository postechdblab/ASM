dir=meta_models

mkdir -p $dir

log=$dir/train_stack.log

python run_experiment.py --dataset stack --generate_models --data_path datasets/stack/{}.csv --model_path $dir > $log
