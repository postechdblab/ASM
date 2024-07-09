"""Registry of datasets and schemas."""
import collections
import os
import pickle

import numpy as np
import pandas as pd

import collections
from common import CsvTable

dataset_list = ['imdb', 'stack', 'stats', 'stack_ss100']

def LoadImdb(table=None,
             data_dir='./datasets/job_csv_export/',
             try_load_parsed=True,
             use_cols='simple',
             irr_attrs=[],
             PK_tuples_np=None,
             all_dvs=None
             ):

    def TryLoad(table_name, filepath, use_cols, irr_attrs, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table37'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table37'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            irr_attrs=irr_attrs,
            PK_tuples_np=PK_tuples_np,
            all_dvs=all_dvs,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        return None

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            irr_attrs=irr_attrs,
            type_casts={},
        )
        return table
    assert False

def LoadDataset(dataset, table, use_cols, data_dir, try_load_parsed=True, pre_add_noise=False, pre_normalize=False, post_normalize=True, irr_attrs=[], PK_tuples_np=None, all_dvs=None):
    if dataset in ['imdb', 'stack', 'stats', 'stack_ss100']:
        return LoadImdb(table, data_dir=data_dir,try_load_parsed=try_load_parsed, use_cols=use_cols, irr_attrs=irr_attrs, PK_tuples_np=PK_tuples_np, all_dvs=all_dvs)
    assert False
