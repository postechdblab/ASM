import argparse
import logging
import os
import time
import shutil
import numpy as np
import pandas as pd
import cProfile, pstats

from Evaluation.training import train
from Evaluation.testing import test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')

    parser.add_argument('--generate_models', action='store_true')
    parser.add_argument('--data_path', default='../datasets/imdb/{}.csv')
    parser.add_argument('--ar_path', default='')
    parser.add_argument('--config_path', default='')
    parser.add_argument('--model_path', default='meta_models/')

    parser.add_argument('--sample_size', type=int, default=None)

    parser.add_argument('--query_name', type=str, default=None)

    parser.add_argument('--evaluate', help='Evaluates models to compute cardinality bound', action='store_true')
    parser.add_argument('--query_file', default='job_queries/all_queries.pkl')
    parser.add_argument('--query_sub_plan_file', default='job_queries/all_sub_plan_queries_str.pkl')
    parser.add_argument('--query_predicate_location', type=str, default=None)
    parser.add_argument('--output_predicate_location', type=str, default=None)
    parser.add_argument('--save_folder', default='job_CE')

    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if args.dataset in ['imdb', 'stack', 'stats', 'stack_ss100']:
        if args.generate_models:
            bound_ensemble = train(args.data_path, args.model_path, args.dataset)
        elif args.evaluate:
            test(args.model_path,
                 args.query_file,
                 args.query_sub_plan_file,
                 args.query_name,
                 args.save_folder,
                 args.ar_path,
                 args.config_path,
                 args.sample_size,
                 args.query_predicate_location,
                 args.dataset,
                 args.output_predicate_location)
        else:
            assert False
    else:
        assert False
