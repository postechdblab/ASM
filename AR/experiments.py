"""Experiment configurations.

EXPERIMENT_CONFIGS holds all registered experiments.

TEST_CONFIGS (defined at end of file) stores "unit tests": these are meant to
run for a short amount of time and to assert metrics are reasonable.

Experiments registered here can be launched using:

  >> python run.py --run <config> [ <more configs> ]
  >> python run.py  # Runs all tests in TEST_CONFIGS.
"""
import os

from ray import tune

EXPERIMENT_CONFIGS = {}
TEST_CONFIGS = {}

# Common config. Each key is auto set as an attribute (i.e. NeuroCard.<attr>)
# so try to avoid any name conflicts with members of that class.

BASE_CONFIG = {
    'cwd': os.getcwd(),
    'epochs_per_iteration': 1,
    'num_eval_queries_per_iteration': 100,
    'num_eval_queries_at_end': 2000,
    'num_eval_queries_at_checkpoint_load': 10000,
    'epochs': 10,
    'seed': None,
    'order_seed': None,
    'bs': 2048,
    'order': None,
    'layers': 2,
    'fc_hiddens': 128,
    'warmups': 1000,
    'constant_lr': None,
    'lr_scheduler': None,
    'custom_lr_lambda': None,
    'optimizer': 'adam',
    'residual': True,
    'direct_io': True,
    'input_encoding': 'embed',
    'output_encoding': 'embed',
    'query_filters': [5, 12],
    'force_query_cols': None,
    'embs_tied': True,
    'embed_size': 32,
    'input_no_emb_if_leq': True,
    'resmade_drop_prob': 0.,

    'use_data_parallel': False,

    'checkpoint_to_load': None,
    'disable_learnable_unk': False,
    'per_row_dropout': True,
    'dropout': 1,
    'table_dropout': False,
    'fixed_dropout_ratio': False,
    'asserts': None,
    'special_orders': 0,
    'special_order_seed': 0,
    'join_tables': [],
    'label_smoothing': 0.0,
    'compute_test_loss': False,

    'factorize': False,
    'factorize_blacklist': None,
    'grouped_dropout': True,
    'subvar_dropout': False,
    'adjust_fact_col': False,
    'factorize_fanouts': False,

    'eval_psamples': [100, 1000, 10000],
    'eval_join_sampling': None,

    'use_transformer': False,
    'transformer_args': {},

    'save_checkpoint_at_end': True,
    'checkpoint_every_epoch': True,

    '_save_samples': None,
    '_load_samples': None,
    'num_orderings': 1,
    'num_dmol': 0,

    'mode' : 'TRAIN',
    'save_eval_result' : True,
    'rust_random_seed' : 0,
    'data_dir': 'datasets/imdb/',
    'epoch' :0,
    'sep' : '#',
    'verbose_mode' : False,
    'accum_iter' : 1,

    'activation': 'relu',
    'num_sample_per_tuple' : 1,

    'num_sigmoids': 0,
    'pre_add_noise': False,
    'post_add_noise': False,
    'pre_normalize': False,
    'post_normalize': False,
    'learnable_unk': False,

    'sample_alg': 'ps',
    'queries_csv': None,

    'PK_tuples_np_loc': None,
}

JOB_LIGHT_BASE = {
    'dataset': 'imdb',
    'join_clauses': None,
    'join_how': 'outer',
    'join_name': 'job-light',
    'use_cols': 'simple',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    'epochs': 1,
    'bs': 2048,
    'max_steps': 500,
    'warmups': 0.05,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 4,
    'layers': 4,
    'compute_test_loss': True,
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 70,
    'eval_psamples': [4000],
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,
}

JOB_M = {
    'epochs': 10,
    'bs': 1000,
    'resmade_drop_prob': 0.1,
    'max_steps': 1000,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 16,
    'warmups': 0.15,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 113,
    'eval_psamples': [1000],
}

JOB_M_FACTORIZED = {
    'factorize': True,
    'factorize_blacklist': [],
    'factorize_fanouts': True,
    'word_size_bits': 14,
    'bs': 2048,
    'max_steps': 512,
    'epochs': 20,
    'checkpoint_every_epoch': True,
    'epochs_per_iteration': 1,
}

IMDB_FULL = {
    'join_tables': ['title', 'aka_title', 'movie_link', 'cast_info', 'movie_info', 'movie_info_idx', 'kind_type', 'movie_keyword', 'movie_companies', 'complete_cast', 'link_type', 'char_name', 'role_type', 'name', 'info_type', 'keyword', 'company_name', 'company_type', 'comp_cast_type', 'aka_name', 'person_info'],
}


SYN_SINGLE = {
    'join_tables': ['table0'],
    'join_keys': {},
    'join_root': 'table0',
    'join_clauses': [],
    'use_cols': 'single',
    'join_how': 'outer',
    'dataset': 'synthetic',
    'join_name' : 'syn_single',
    'eval_psamples': [512],
    'sep' :'#',
    'table_dropout' : False,
    'num_eval_queries_at_checkpoint_load': 10000,
    'learnable_unk': True,
}

SYN_SINGLE_TUNE = {
    'epochs': 40,
    'max_steps' : 512,
    'use_data_parallel':False,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration':0,
    'num_eval_queries_at_end':0,
    'fc_hiddens': 2048,
    'layers' : 4,
    'embed_size' : 32,
    'word_size_bits': 14,
    'checkpoint_every_epoch' : False,
}

### EXPERIMENT CONFIGS ###
EXPERIMENT_CONFIGS = {
    'syn-single-00-dist': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'join_name': 'syn-single-00',
    }),
}

for table in IMDB_FULL['join_tables']:
    EXPERIMENT_CONFIGS[f'imdb-single-{table}'] = dict(EXPERIMENT_CONFIGS['syn-single-00-dist'])

    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['fc_hiddens'] = 128
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['epochs'] = 20

    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['checkpoint_every_epoch'] = True
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['save_checkpoint_at_end'] = False

    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['eval_psamples'] = [2048]
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['dataset'] = 'imdb'
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['data_dir'] = f'datasets/imdb/{table}/'
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['use_cols'] = None
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['PK_tuples_np_loc'] = f'datasets/imdb/{table}/tuples_np.pkl'
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['subvar_dropout'] = True
    EXPERIMENT_CONFIGS[f'imdb-single-{table}']['adjust_fact_col'] = True

    if "_type" in table:
        EXPERIMENT_CONFIGS[f'imdb-single-{table}']['fc_hiddens'] = 16
        EXPERIMENT_CONFIGS[f'imdb-single-{table}']['sampler_batch_size'] = 32
        EXPERIMENT_CONFIGS[f'imdb-single-{table}']['constant_lr'] = 0.01

    TEST_CONFIGS[f'imdb-single-{table}_infer'] = dict(EXPERIMENT_CONFIGS[f'imdb-single-{table}'])
    TEST_CONFIGS[f'imdb-single-{table}_infer']['queries_csv'] = ""

for table in ['account', 'answer', 'question', 'site', 'so_user', 'tag', 'tag_question']:
    EXPERIMENT_CONFIGS[f'stack-single-{table}'] = dict(EXPERIMENT_CONFIGS['syn-single-00-dist'])
    if table == "site":
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['fc_hiddens'] = 128
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['layers'] = 2
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['warmups'] = 1
    if table == 'tag':
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['per_row_dropout'] = False
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['dropout'] = 0
        EXPERIMENT_CONFIGS[f'stack-single-{table}']['learnable_unk'] = False
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['checkpoint_every_epoch'] = True
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['save_checkpoint_at_end'] = False

    EXPERIMENT_CONFIGS[f'stack-single-{table}']['eval_psamples'] = [2048]
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['dataset'] = 'stack'

    EXPERIMENT_CONFIGS[f'stack-single-{table}']['data_dir'] = f'datasets/stack/{table}/'
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['use_cols'] = None
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['PK_tuples_np_loc'] = f'datasets/stack/{table}/tuples_np.pkl'
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['subvar_dropout'] = True
    EXPERIMENT_CONFIGS[f'stack-single-{table}']['adjust_fact_col'] = True

    TEST_CONFIGS[f'stack-single-{table}_infer'] = dict(EXPERIMENT_CONFIGS[f'stack-single-{table}'])
    TEST_CONFIGS[f'stack-single-{table}_infer']['queries_csv'] = ""


for table in ['badges', 'comments', 'postHistory', 'postLinks', 'posts', 'tags', 'users', 'votes']:
    EXPERIMENT_CONFIGS[f'stats-single-{table}'] = dict(EXPERIMENT_CONFIGS['syn-single-00-dist'])
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['fc_hiddens'] = 32
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['embed_size'] = 5
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['layers'] = 3
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['epochs'] = 7
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['sampler_batch_size'] = 1024 * 4

    EXPERIMENT_CONFIGS[f'stats-single-{table}']['checkpoint_every_epoch'] = True
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['save_checkpoint_at_end'] = False

    EXPERIMENT_CONFIGS[f'stats-single-{table}']['eval_psamples'] = [2048]
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['dataset'] = 'stats'
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['data_dir'] = f'datasets/stats/{table}/'
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['use_cols'] = None

    EXPERIMENT_CONFIGS[f'stats-single-{table}']['PK_tuples_np_loc'] = f'datasets/stats/{table}/tuples_np.pkl'
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['subvar_dropout'] = True
    EXPERIMENT_CONFIGS[f'stats-single-{table}']['adjust_fact_col'] = True

    TEST_CONFIGS[f'stats-single-{table}_infer'] = dict(EXPERIMENT_CONFIGS[f'stats-single-{table}'])
    TEST_CONFIGS[f'stats-single-{table}_infer']['queries_csv'] = ""

for name in TEST_CONFIGS:
    TEST_CONFIGS[name].update({'save_checkpoint_at_end': False})
    TEST_CONFIGS[name].update({'mode': 'INFERENCE'})
EXPERIMENT_CONFIGS.update(TEST_CONFIGS)
