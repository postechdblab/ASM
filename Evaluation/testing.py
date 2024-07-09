import pickle5 as pickle
import sys
import time
import os
import numpy as np
import torch
import pandas as pd
import multiprocessing
import copy
from Join_scheme.bound import Bound_ensemble

sys.path.append('AR')

from experiments import EXPERIMENT_CONFIGS
from neurocard import NeuroCard

def load_neurocard(dataset, table, ar_path, config_path):

    config = EXPERIMENT_CONFIGS[config_path.format(dataset, table)]
    config['__run'] = config['workload_name'] = config_path.format(dataset, table)
    if os.path.exists(ar_path.format(dataset, table)):
        print('table', table, 'updated')
        config['checkpoint_to_load'] = ar_path.format(dataset, table)
    else:
        print('table', table, 'not updated')
        config['checkpoint_to_load'] = f"AR_models/{dataset}-single-{table}.tar"
    config['queries_csv'] = None
    config['__gpu'] = 1
    config['__cpu'] = 1
    config['external'] = False
    nc = NeuroCard(config)

    min_count = pickle.load(open(f"{config['data_dir']}/min_count.pkl","rb"))
    nc.min_count = dict()
    for col in min_count:
        assert np.all(min_count[col] >= 1.), f'{np.where(min_count[col]) < 1.}'
        nc.min_count[col] = torch.tensor(min_count[col], device=nc.get_device())

    return nc

def test(model_path,
         query_file,
         query_sub_plan_file,
         query_name,
         save_res,
         ar_path,
         config_path,
         sample_size,
         query_predicate_location,
         dataset,
         output_predicate_location
         ):

    bound_ensemble = None

    with open(model_path, "rb") as f:
        bound_ensemble_nc = pickle.load(f)

    ncs = dict()
    for table_obj in bound_ensemble_nc.schema.tables:
        table = table_obj.table_name

        nc = load_neurocard(dataset, table, ar_path, config_path)
        use_raw_table = table_obj.table_size < 1000
        nc.ready_for_evaluate(use_raw_table)

        ncs[table] = nc

    bound_ensemble_nc.ncs = ncs

    bound_ensemble_nc.use_ar = True
    if sample_size is None:
        bound_ensemble_nc.sample_size = 2048
    else:
        bound_ensemble_nc.sample_size = sample_size

    with open(query_file, "rb") as f:
        all_queries = pickle.load(f)
    with open(query_sub_plan_file, "rb") as f:
        all_sub_plan_queries_str = pickle.load(f)

    # total initialize model and sample time
    total_q_time = 0.
    total_m_time = 0.
    total_s_time = 0.

    res = dict()
    q_times = dict()
    m_times = dict()
    s_times = dict()

    t = time.time()

    for i, q_name in enumerate(all_queries):
        if query_name is not None:
            if q_name != query_name:
                continue

        print(f'query: {q_name}')

        if os.path.exists(query_predicate_location + '/' + q_name + '.pkl'):
            with open(query_predicate_location + '/' + q_name + '.pkl', "rb") as f:
                bound_ensemble_nc.query_predicate = pickle.load(f)
        else:
            continue

        if True:
            if save_res:
                if os.path.exists(save_res + '.' + q_name):
                    continue
            torch.manual_seed(0)

            q_time_start = time.time()
            ests, c_time, m_time, s_time = bound_ensemble_nc.get_cardinality_bound_all(
                all_queries[q_name],
                all_sub_plan_queries_str[q_name]
            )

            q_time = time.time() - q_time_start - c_time + m_time
            assert c_time >= m_time
            total_q_time += q_time
            total_m_time += m_time
            total_s_time += s_time
            q_times[q_name] = q_time
            m_times[q_name] = m_time
            s_times[q_name] = s_time
            if save_res:
                f_query = open(save_res + '.' + q_name, "w")
                for est in ests:
                    f_query.write(str(est) + "\n")
                f_query.close()
            res[q_name] = ests

            if dataset == 'stack':
                from stack_utils import qname_to_qindex
            elif dataset == 'imdb':
                from imdb_utils import qname_to_qindex
            elif dataset == 'stats':
                from stats_utils import qname_to_qindex
            else:
                assert False

            df = pd.DataFrame(columns = ['id', 'name', 'q_time', 'm_time', 's_time'])
            for q_name in res:
                df = df.append({'id': qname_to_qindex(q_name), 'name': q_name,
                                'q_time': q_times[q_name], 'm_time': m_times[q_name], 's_time': s_times[q_name]},
                               ignore_index = True)
            df = df.sort_values(by=['id'])
            df.to_csv(save_res + f'.time.csv')

    print("total estimation latency is: ", total_q_time)
    print("total initialize model latency is: ", total_m_time)
    print("total initialize sample latency is: ", total_s_time)
