import copy
import logging
import pickle
import numpy as np
import pandas as pd
import time

from Schemas.imdb.schema import gen_imdb_schema
from Schemas.stack.schema import gen_stack_schema
from Schemas.stats.schema import gen_stats_light_schema

logger = logging.getLogger(__name__)

def identify_key_values(schema):
    all_keys = set()
    equivalent_keys = dict()
    for i, join in enumerate(schema.relationships):
        keys = join.identifier.split(" = ")
        all_keys.add(keys[0])
        all_keys.add(keys[1])
        seen = False
        for k in equivalent_keys:
            if keys[0] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[1])
                seen = True
                break
            elif keys[1] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[0])
                seen = True
                break
        if not seen:
            equivalent_keys[keys[-1]] = set(keys)

    assert len(all_keys) == sum([len(equivalent_keys[k]) for k in equivalent_keys])
    return all_keys, equivalent_keys



def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def read_table_hdf(table_obj):
    """
    Reads hdf from path, renames columns and drops unnecessary columns
    """
    df_rows = pd.read_hdf(table_obj.csv_file_location)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def read_table_csv(table_obj, csv_seperator=',', stats=True):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    if stats:
        df_rows = pd.read_csv(table_obj.csv_file_location)
    else:
        df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', encoding='utf-8',
                              quotechar='"',
                              sep=csv_seperator)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def get_imdb_schema(data_path, dataset):
    if 'stack' in dataset:
        schema = gen_stack_schema(data_path)
    elif dataset == 'imdb':
        schema = gen_imdb_schema(data_path)
    elif dataset == 'stats':
        schema = gen_stats_light_schema(data_path)
    all_keys, equivalent_keys = identify_key_values(schema)
    return schema, all_keys, equivalent_keys
