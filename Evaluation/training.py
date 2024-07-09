import pickle
import pandas as pd
import time
import numpy as np
import os
from Join_scheme.join_graph import dfs, get_PK
from Join_scheme.data_prepare import get_imdb_schema
from Join_scheme.bound import Bound_ensemble
import psycopg2

use_given_order_PKs = True
compute_tuples_np = True

def train(data_path, model_folder, dataset):

    if True:
        schema, all_keys, equivalent_keys = get_imdb_schema(data_path, dataset)

        all_df_rows = dict()

        # make PK_tuples_np files
        all_tuples_np = dict()
        for table_obj in schema.tables:
            table = table_obj.table_name
            all_tuples_np[table] = dict()

        table_size = dict()

        if "stack" in dataset:
            # reading from csv is faulty
            conn = psycopg2.connect(database=dataset, user="postgres", password="postgres", host="localhost", port=5432,)
            conn.set_client_encoding('UTF8')

        P_tuples_np_time = 0

        for PK in equivalent_keys:
            [P_table, P_col] = PK.split('.')
            table_obj = schema.table_dictionary[P_table]

            if "stack" in dataset:
                query = "select * from " + table_obj.table_name + " ;"
                df_rows = pd.read_sql(query, conn)
            else:
                if os.path.exists(table_obj.csv_file_location):
                    print("reading", table_obj.csv_file_location)
                    if dataset == 'imdb':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['']
                                            )
                    elif dataset == 'stats':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=[''],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = ','
                                      )
                else:
                    continue

            df_rows.columns = table_obj.attributes

            table_size[PK] = len(df_rows)
            all_df_rows[P_table] = df_rows
            if compute_tuples_np:
                P_tuples_np_time_start = time.time()
                P_tuples_np = np.unique(pd.concat([pd.Series([np.nan, 0]), df_rows[P_col]]).values)
                P_tuples_np_time += time.time() - P_tuples_np_time_start
                assert np.any(pd.isnull(P_tuples_np))

                for FK in equivalent_keys[PK]:
                    [table, col] = FK.split('.')

                    all_tuples_np[table][col] = P_tuples_np
                    print(f"FK {FK} tuples_np of PK {PK} = {P_tuples_np}")

        dfs_PKs_time_start = time.time()
        # set DFS order of PKs
        if use_given_order_PKs:
            if 'stack' in dataset:
                dfs_PKs = ['site.site_id', 'question.id', 'tag.id', 'so_user.id', 'account.id']
            elif dataset == 'imdb':
                dfs_PKs = ['title.id', 'kind_type.id', 'comp_cast_type.id', 'info_type.id', 'name.id', 'char_name.id', 'role_type.id', 'link_type.id', 'keyword.id', 'company_name.id', 'company_type.id']
            elif dataset == 'stats':
                dfs_PKs = ['users.Id', 'posts.Id']
            else:
                assert False
        else:
            # perform DFS
            # weird, but the result of DFS changes for each trial
            # for such a case, just use the above DFS order
            if 'stack' in dataset:
                start_PK = 'site.site_id'
            elif dataset == 'imdb':
                start_PK = 'title.id'
            elif dataset == 'stats':
                start_PK = 'users.Id'
            else:
                assert False
            dfs_PKs = dfs(start_PK, schema, equivalent_keys, None, [])

        dfs_PKs_time = time.time() - dfs_PKs_time_start
        print('dfs PKs or order of PKs:', dfs_PKs)

        # create meta model
        be = Bound_ensemble(schema)
        be.all_dfs_PKs = dfs_PKs

        from AR.experiments import EXPERIMENT_CONFIGS

        be.reordered_attributes = dict()

        reorder_time = 0

        for table_obj in schema.tables:
            table = table_obj.table_name

            loc = table_obj.csv_file_location
            attrs = table_obj.attributes
            irr_attrs = table_obj.irrelevant_attributes
            print('table', table, 'loc', loc, 'attrs', attrs, 'irr attrs', irr_attrs)

            reorder_time_start = time.time()

            order = []
            temp_orders = []
            j = 0
            for attr in attrs:
                if attr not in irr_attrs:
                    PK = get_PK(table + "." + attr, equivalent_keys)
                    if PK is not None:
                        order.append(-1)
                        temp_orders.append(dfs_PKs.index(PK))
                    else:
                        order.append(j)
                        j += 1

            global_orders = sorted(list(range(len(temp_orders))), key=lambda x: temp_orders[x])

            reverse_order = [-1] * len(global_orders)
            for i, k in enumerate(global_orders):
                reverse_order[k] = i

            # make the join order consistent with the dfs_PKs
            PK_i = 0
            for i, k in enumerate(order):
                if k == -1:
                    order[i] = j + reverse_order[PK_i]
                    print(f'order[{i}] = {j + reverse_order[PK_i]}')
                    #j += 1
                    PK_i += 1

            reverse_order = [-1] * len(order)
            for i, k in enumerate(order):
                reverse_order[k] = i

            reorder_time += time.time() - reorder_time_start

            table_dir = data_path.format(table)[:-4]
            table_path = table_dir + '/table0.csv'

            if True:
                os.makedirs(table_dir, exist_ok=True)

                if table in all_df_rows:
                    df_rows = all_df_rows[table]
                else:
                    if dataset == 'stack':
                        query = "select * from " + table_obj.table_name + " ;"
                        print(query)
                        df_rows = pd.read_sql(query, conn)
                    elif dataset == 'imdb':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['']
                                            )
                    elif dataset == 'stats':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=[''],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = ','
                                      )
                    
                    df_rows.columns = table_obj.attributes

                reorder_time_start = time.time()

                for attr in irr_attrs:
                    df_rows = df_rows.drop(attr, axis=1)
                df_rows = df_rows.iloc[:, reverse_order]

                be.reordered_attributes[table] = list(df_rows.columns)

                reorder_time += time.time() - reorder_time_start

            if not os.path.exists(table_path):
                if 'stack' in dataset:
                    df_rows[0:0].to_csv(table_path, header=True, index=False)
                else:
                    df_rows.to_csv(table_path, header=True, index=False)

            if compute_tuples_np:
                assert table in all_tuples_np, f'table {table}'
                
                tuples_np_path = table_dir + '/tuples_np.pkl'
                if not os.path.exists(tuples_np_path):
                    pickle.dump(all_tuples_np[table], open(tuples_np_path, 'wb'))

        # save model
        model_path = model_folder + f"/model_{dataset}.pkl"
        if 'stack' in dataset:
            conn.close()
        if not os.path.exists(model_path):
            pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"models save at {model_path}")

        print('P_tuples_np_time:', P_tuples_np_time)
        print('dfs_PKs_time:', dfs_PKs_time)
        print('reorder_time:', reorder_time)
        print('total time:', P_tuples_np_time + dfs_PKs_time + reorder_time)
