import numpy as np
import copy
import os
import pickle5 as pickle
import math
import re
import json
import torch
import time
import cProfile, pstats
import pandas.io.sql as sqlio

from Join_scheme.join_graph import get_join_hyper_graph, parse_query_all_join, dfs, get_sub_query_equivalent_group
from Join_scheme.data_prepare import identify_key_values
from logical_tree import parse_logic_tree, fillcol, get_subtree, to_neurocard_ops

no_PE = False

def get_org_key(key, tables_all):
    [a, c] = key.split(".")
    if a in tables_all:
        return tables_all[a] + "." + c
    return key

class NCFactor:

    def __init__(self, table, table_len, densities, idx, sample, logits,
                 cols, ops, vals, dom_ops
                 ):
        self.table = table
        self.table_len = table_len

        self.densities = densities
        self.densities = self.densities.view(-1,1)
        self.cur_idx = idx
        self.sample = sample
        self.logits = logits
        self.cols = cols
        self.ops = ops
        self.vals = vals
        self.dom_ops = dom_ops

        self.PK_to_idx = dict()
        self.P = dict()
        self.min_count = dict()
        self.min_PK_order = 1000000

    def set_PK_to_idx(self, nc, table_key_group):
        for key in table_key_group:
            PK = table_key_group[key]
            idx = nc.get_natural_idx(key.split(".")[-1])
            self.PK_to_idx[PK] = idx

class NCFactor_Group:
    def __init__(self, table_len):
        self.table_len = table_len

        self.join_cond = None
        self.equivalent_groups = None
        self.table_key_equivalent_group = None

def initialize_model(ncs, tables_alias, join_keys, schema,
                     sample_size,
                     data):
    conditional_factors = dict()
    m_time = 0.

    def evaluate_model(i, alias, table, pred, keys):
        nc = ncs[table]

        query = dict()

        if len(pred) > 0:
            tree = parse_logic_tree(pred, alias, table, schema)

            cols = fillcol(tree)

            to_neurocard_ops(tree)
            for col in cols:
                subtree = get_subtree(tree, col)
                query[col] = subtree

        e_time_start = time.time()
        p, table_len, cur_idx, sample, logits, cols, ops, vals, dom_ops = nc.evaluate_one_tree(query, keys, sample_size)
        if not torch.is_tensor(p):
            p = torch.Tensor(p).to(nc.get_device())
        e_time = time.time() - e_time_start

        return e_time, NCFactor(alias,
                                table_len,
                                p,
                                cur_idx,
                                sample,
                                logits,
                                cols,
                                ops,
                                vals,
                                dom_ops
                                )
    
    for i, alias in enumerate(data):
        (table, pred, keys) = data[alias]
        e_time, temp = evaluate_model(i, alias, table, pred, keys)
        conditional_factors[alias] = temp
        m_time += e_time

    return conditional_factors, m_time


class Bound_ensemble:
    """
    This the class where we store all the trained models and perform inference on the bound.
    """

    def __init__(self, schema):
        self.schema = schema
        self.all_keys, self.equivalent_keys = identify_key_values(schema)
        self.all_join_conds = None

    def parse_query_simple(self, query):
        """
        If your selection query contains no aggregation and nested sub-queries, you can use this function to parse a
        join query. Otherwise, use parse_query function.
        """
        tables_all, join_cond, join_keys = parse_query_all_join(query)
        table_filters = dict()
        return tables_all, table_filters, join_cond, join_keys

    def get_all_id_conditional_distribution(self, tables_alias, join_keys):
        t = time.time()

        nc_factors, m_time = initialize_model(self.ncs, tables_alias, join_keys, self.schema, sample_size=self.sample_size, data=self.query_predicate)

        return nc_factors, time.time() - t, m_time

    def get_col_index(self, key, tables_all):
        [a, c] = key.split(".")
        t = tables_all[a]
        return self.reordered_attributes[t].index(c)

    def initialize_sample(self, factors, tables_all, equivalent_group, table_key_group_map):
        t = time.time()

        for PK in equivalent_group:
            table_PK = get_org_key(PK, tables_all)
            order = self.all_dfs_PKs.index(table_PK)
            for FK in equivalent_group[PK]:
                [alias, column] = FK.split('.')
                factors[alias].min_PK_order = min(factors[alias].min_PK_order, order)

        self.query_dfs_PKs = sorted(list(equivalent_group.keys()),
                                    key=lambda PK: (self.all_dfs_PKs.index(get_org_key(PK, tables_all)),
                                                    sorted([(factors[FK.split('.')[0]].min_PK_order, self.get_col_index(FK, tables_all))
                                                            for FK in equivalent_group[PK]]
                                                           )
                                                    )
                                    )
        if len(self.query_dfs_PKs) > 2:
            self.sum_P_prod = True
        else:
            self.sum_P_prod = False

        for alias in factors:
            table = tables_all[alias]
            factors[alias].set_PK_to_idx(self.ncs[table], table_key_group_map[alias])
            factors[alias].min_PK_order = 100000
            factors[alias].sort_key = []

        for PK in equivalent_group:
            order = self.query_dfs_PKs.index(PK)
            for FK in equivalent_group[PK]:
                [alias, column] = FK.split('.')
                factors[alias].min_PK_order = min(factors[alias].min_PK_order, order)

        self.num_fact_cols = dict()
        self.fact_domains = dict()

        for order, PK in enumerate(self.query_dfs_PKs):
            self.num_fact_cols[PK] = None
            self.fact_domains[PK] = None

            for key in equivalent_group[PK]:
                [alias, column] = key.split('.')
                table = tables_all[alias]
                idxs = factors[alias].PK_to_idx[PK]
                assert len(idxs) > 0
                if self.num_fact_cols[PK] is not None:
                    assert self.fact_domains[PK]
                    assert self.num_fact_cols[PK] == len(idxs), f'prev # fact cols = {self.num_fact_cols[PK]} but new # fact cols = {len(idxs)}'
                    for i, idx in enumerate(idxs):
                        domain = self.ncs[table].get_domain(idx)
                else:
                    self.num_fact_cols[PK] = len(idxs)
                    self.fact_domains[PK] = []

                    for i, idx in enumerate(idxs):
                        domain = self.ncs[table].get_domain(idx)
                        self.fact_domains[PK].append(domain)

                if len(idxs) > 1:
                    factors[alias].min_count[PK] = self.ncs[table].min_count[column]
                    assert torch.all(factors[alias].min_count[PK] > 0)

                factors[alias].sort_key.append(order)

        for alias in factors:
            assert factors[alias].min_PK_order < 1000000

        tables_in_order = sorted(factors.keys(), key=lambda alias: factors[alias].sort_key)
        self.table_orders = {alias: tables_in_order.index(alias) for alias in factors}

        self.inv_P = dict()

        for order, PK in enumerate(self.query_dfs_PKs):
            self.inv_P[PK] = []

            for i in range(self.num_fact_cols[PK]):
                if i != 0:
                    continue
                P_prod = None

                for key in equivalent_group[PK]:
                    [alias, column] = key.split('.')
                    factor = factors[alias]
                    table = tables_all[alias]
                    idx = factor.PK_to_idx[PK][i]

                    assert factor.cur_idx <= idx, f'factor {alias} cur_idx {factor.cur_idx} idx {idx} PK_to_idx {factor.PK_to_idx}'

                    P, factor.sample, factor.logits, returned_idx = self.ncs[table].get_P(factor.cur_idx, idx, self.sample_size, factor.sample, factor.logits,
                                                                                          factor.cols, factor.ops, factor.vals, factor.dom_ops
                                                                                          )
                    assert returned_idx == idx

                    if self.ncs[table].use_raw_table:
                        assert len(P.shape) == 1
                        P[0] = 0.
                        P = P.view(1,-1).expand(self.sample_size, -1)
                    else:
                        assert len(P.shape) == 2
                        mask = torch.ones_like(P)
                        mask[:,0] = 0.
                        P = P * mask

                    assert idx not in factor.P
                    factor.P[idx] = P

                    if P_prod is None:
                        P_prod = P.detach().clone()
                    else:
                        assert P_prod.shape == P.shape, f'P_prod {P_prod.shape} P {P.shape}'
                        if self.sum_P_prod:
                            P_prod = P_prod + P
                        else:
                            P_prod = torch.mul(P_prod, P)

                # mean over the distributions of different samples
                P_prod = P_prod.mean(0, keepdim=True)

                probs_i_summed = P_prod.sum(1, keepdim=True)
                paths_vanished = (probs_i_summed <= 0)
                probs_i_summed = probs_i_summed.masked_fill_(paths_vanished, 1.0/P_prod.shape[-1])

                P_prod = P_prod.div(probs_i_summed)
                assert torch.allclose(P_prod.sum(1, keepdim=True), torch.ones_like(probs_i_summed)), f'{P_prod}, {P_prod.sum(1)}'
                assert not torch.any(paths_vanished), paths_vanished

                samples_i = torch.multinomial(P_prod,
                                              num_samples=self.sample_size,
                                              replacement=True)

                data_to_encode = samples_i.view(-1,1)

                temp_P = torch.gather(P_prod.view(-1,1), 0, data_to_encode)
                inv_P = torch.reciprocal(temp_P)
                inv_P = inv_P.masked_fill_(paths_vanished, 0.0)

                assert data_to_encode.shape == (self.sample_size, 1)
                assert inv_P.shape == data_to_encode.shape

                for key in equivalent_group[PK]:
                    [alias, column] = key.split('.')
                    factor = factors[alias]
                    table = tables_all[alias]
                    idx = factor.PK_to_idx[PK][i]

                    factor.sample = self.ncs[table].put_sample(data_to_encode, factor.sample, idx)

                    assert data_to_encode.shape[0] == self.sample_size
                    assert factor.P[idx].shape[1] == P_prod.shape[1]

                    factor.P[idx] = torch.gather(factor.P[idx], 1, data_to_encode)

                    assert factor.P[idx].shape == data_to_encode.shape

                    if PK in factor.min_count:
                        assert torch.max(data_to_encode) < factor.min_count[PK].shape[-1], f'max data_to_encode {torch.max(data_to_encode)} >= len(min_count) {factor.min_count[PK].shape[-1]}'
                        factor.min_count[PK] = torch.gather(factor.min_count[PK].view(-1,1), 0, data_to_encode)
                        assert factor.min_count[PK].shape == data_to_encode.shape
                        assert torch.all(factor.min_count[PK] >= 1.)

                    factor.cur_idx = idx + 1

                self.inv_P[PK].append(inv_P)

        for alias in factors:
            factor = factors[alias]
            table = tables_all[alias]
            for idx in range(factor.cur_idx, len(factor.cols)):
                col = factor.cols[idx]
                if factor.ops[idx] is not None:
                    if col.factor_id is not None and col.factor_id != 0:
                        continue
                    P, factor.sample, factor.logits, returned_idx = self.ncs[table].get_P(factor.cur_idx,
                                                                                          idx,
                                                                                          self.sample_size,
                                                                                          factor.sample,
                                                                                          factor.logits,
                                                                                          factor.cols,
                                                                                          factor.ops,
                                                                                          factor.vals,
                                                                                          factor.dom_ops
                                                                                          )
                    assert returned_idx == idx

                    assert len(P.shape) == 2
                    assert P.shape[0] == self.sample_size

                    if factor.densities.shape[0] != self.sample_size:
                        factor.densities = factor.densities.expand(self.sample_size, -1)
                    factor.densities = factor.densities * P.sum(-1, keepdim=True)

                    factor.cur_idx = idx + 1


        return time.time() - t

    def get_cardinality_one_nc(self, factors, tables_all, equivalent_group):
        assert len(self.query_dfs_PKs) == len(equivalent_group)

        res = None
        for order, PK in enumerate(self.query_dfs_PKs):
            for i in range(self.num_fact_cols[PK]):
                if i != 0:
                    assert res is not None
                    scale = None
                    for key in equivalent_group[PK]:
                        [alias, column] = key.split('.')
                        table = tables_all[alias]
                        factor = factors[alias]

                        if scale is None:
                            scale = factor.min_count[PK]
                        else:
                            scale = torch.maximum(scale, factor.min_count[PK])
                        res = res / scale
                    continue

                # /P^-1
                if res is None:
                    res = self.inv_P[PK][i]
                else:
                    res = res * self.inv_P[PK][i]

                for key in equivalent_group[PK]:
                    [alias, column] = key.split('.')
                    table = tables_all[alias]
                    factor = factors[alias]
                    idx = factor.PK_to_idx[PK][i]

                    # * P_T
                    res = res * factor.P[idx]

        # * |T|
        for alias in factors:
            if res is None:
                res = factors[alias].table_len
            else:
                res = res * factors[alias].table_len
            res = res * factors[alias].densities

        res = torch.mean(res).item()
        return res


    def decompose_query(self, cached_sub_queries, cand_tables,
                        left_table=None,
                        conditional_factors=None, join_cond=None,
                        equivalent_group=None,
                        ):
        assert not no_PE
        assert left_table is None
        res = []
        sort_key = []

        def decompose(t):
            key = cand_tables.copy()
            key.remove(t)
            key.sort()
            key_str = " ".join(key)
            if key_str in cached_sub_queries:
                res.append((t, key_str))

        if left_table is None:
            decompose(cand_tables[-1])
        else:
            decompose(left_table)
        assert len(res) == 1

        return res

    def get_cardinality_bound_all(self, query_str, sub_plan_query_str_all):
        """
        Get the cardinality bounds for all sub_plan_queires of a query.
        Note: Due to efficiency, this current version only support left_deep plans (like the one generated by postgres),
              but it can easily support right deep or bushy plans.
        :param query_str: the target query
        :param sub_plan_query_str_all: all sub_plan_queries of the target query,
               it should be sorted by number of the tables in the sub_plan_query
        """
        query_convert_time_start = time.time()

        tables_all, table_queries, join_cond, join_keys = self.parse_query_simple(query_str)

        all_aliases = list(tables_all.keys())
        all_aliases.sort()

        equivalent_group, table_equivalent_group, table_key_equivalent_group, table_key_group_map = \
            get_join_hyper_graph(join_keys, self.equivalent_keys, tables_all, join_cond)

        query_convert_time = time.time() - query_convert_time_start
        conditional_factors, initialize_model_query_convert_time, initialize_model_time = self.get_all_id_conditional_distribution(tables_all, join_keys)
        initialize_model_query_convert_time += query_convert_time

        if not no_PE:
            initialize_sample_time = self.initialize_sample(conditional_factors, tables_all, equivalent_group, table_key_group_map)
        else:
            initialize_sample_time = 0

        cached_sub_queries = dict()

        cardinality_bounds = []
        for i, (left_tables, right_tables) in enumerate(sub_plan_query_str_all):
            assert " " not in left_tables, f"{left_tables} contains more than one tables, violating left deep plan"
            all_tables = right_tables.split(" ") + [left_tables]
            sub_plan_query_list = all_tables.copy()
            sub_plan_query_list.sort()
            sub_plan_query_str = " ".join(sub_plan_query_list)
            if no_PE:
                sub_query_factors = dict()
                for alias in all_tables:
                    sub_query_factors[alias] = copy.deepcopy(conditional_factors[alias])

                sub_query_equivalent_group = get_sub_query_equivalent_group(sub_plan_query_list, equivalent_group)

                initialize_sample_time += self.initialize_sample(sub_query_factors, tables_all, sub_query_equivalent_group, table_key_group_map)
                res = self.get_cardinality_one_nc(sub_query_factors, tables_all, sub_query_equivalent_group)
                cardinality_bounds.append(res)
                continue

            if " " in right_tables:
                assert right_tables in cached_sub_queries, f"{right_tables} not in cache, input is not ordered"
                right_bound_factor = cached_sub_queries[right_tables]
                # A join (B join C)
                decompositions = self.decompose_query(cached_sub_queries, all_tables,
                                                      conditional_factors=conditional_factors,
                                                      join_cond=join_cond,
                                                      equivalent_group=equivalent_group
                                                      )
                assert len(decompositions) == 1
                for (t, key_str) in decompositions:
                    key_factor = cached_sub_queries[key_str]

                    curr_bound_factor, res = self.join_with_one_table_nc(sub_plan_query_str,
                                                                                        t,
                                                                                        tables_all,
                                                                                        key_factor,
                                                                                        conditional_factors[t],
                                                                                        conditional_factors,
                                                                                        table_equivalent_group,
                                                                                        table_key_equivalent_group,
                                                                                        table_key_group_map,
                                                                                        join_cond
                                                                                        )
            else:
                # A join B
                curr_bound_factor, res = self.join_two_tables_nc(sub_plan_query_str,
                                                                                left_tables,
                                                                                right_tables,
                                                                                tables_all,
                                                                                conditional_factors,
                                                                                join_keys,
                                                                                table_equivalent_group,
                                                                                table_key_equivalent_group,
                                                                                table_key_group_map,
                                                                                join_cond
                                                                                )
            cached_sub_queries[sub_plan_query_str] = curr_bound_factor
            res = max(res, 1)
            cardinality_bounds.append(res)

        return cardinality_bounds, initialize_model_query_convert_time, initialize_model_time, initialize_sample_time

    def join_with_one_table_nc(self, sub_plan_query_str: str, left_table: str, tables_all: dict(), right_bound_factor: NCFactor_Group, cond_factor_left: NCFactor, conditional_factors,
                               table_equivalent_group, table_key_equivalent_group, table_key_group_map, join_cond
                               ):
        equivalent_key_group, union_key_group_set, union_key_group, new_join_cond = \
            self.get_join_keys_with_table_group(left_table, right_bound_factor, tables_all, table_equivalent_group,
                                                table_key_equivalent_group, table_key_group_map, join_cond)

        new_union_key_group = dict()

        res = right_bound_factor.table_len * cond_factor_left.table_len * cond_factor_left.densities

        PKs_in_order = sorted(list(equivalent_key_group.keys()), key=lambda x: self.query_dfs_PKs.index(x))

        for PK in PKs_in_order:
            left_keys = equivalent_key_group[PK]["left"]
            right_keys = equivalent_key_group[PK]["right"]

            assert len(left_keys) == 1
            assert len(right_keys) >= 1

            new_union_key_group[PK] = [PK]

            left_idxs = cond_factor_left.PK_to_idx[PK]

            if len(right_keys) == 1:

                [right_table, right_key] = right_keys[0].split(".")

                assert right_table in conditional_factors

                cond_factor_right = conditional_factors[right_table]
                right_idxs = cond_factor_right.PK_to_idx[PK]

                for i, right_idx in enumerate(right_idxs):
                    if i != 0:
                        res = res / torch.maximum(cond_factor_left.min_count[PK], cond_factor_right.min_count[PK])
                        continue
                    res = cond_factor_right.P[right_idx] * self.inv_P[PK][i] * res

            for i, left_idx in enumerate(left_idxs):
                if i != 0:
                    if len(right_keys) != 1:
                        res = res / cond_factor_left.min_count[PK]
                    continue
                res = cond_factor_left.P[left_idx] * res

        res = torch.mean(res).item()

        new_factor = NCFactor_Group(res)

        for group in union_key_group:
            if group not in new_union_key_group:
                new_union_key_group[group] = []
            for table, keys in union_key_group[group]:
                for key in keys:
                    if key in equivalent_key_group:
                        continue
                    elif group in equivalent_key_group:
                        continue
                    new_union_key_group[group].append(key)

        new_factor.join_cond = new_join_cond
        new_factor.equivalent_groups = union_key_group_set
        new_factor.table_key_equivalent_group = new_union_key_group

        return new_factor, res

    def get_join_keys_with_table_group(self, left_table, right_bound_factor, tables_all, table_equivalent_group,
                                       table_key_equivalent_group, table_key_group_map, join_cond):
        """
            Get the join keys between two tables
        """

        actual_join_cond = []
        for cond in join_cond[left_table]:
            if cond in right_bound_factor.join_cond:
                actual_join_cond.append(cond)
        equivalent_key_group = dict()
        union_key_group_set = table_equivalent_group[left_table].union(right_bound_factor.equivalent_groups)
        union_key_group = dict()
        new_join_cond = right_bound_factor.join_cond.union(join_cond[left_table])
        if len(actual_join_cond) != 0:
            for cond in actual_join_cond:
                key1 = cond.split("=")[0].strip()
                key2 = cond.split("=")[1].strip()
                if key1.split(".")[0] == left_table:
                    key_left = left_table + "." + key1.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["left"].append(key_left)
                    else:
                        equivalent_key_group[key_group]["left"] = [key_left]
                    right_table = key2.split(".")[0]
                    key_right = right_table + "." + key2.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if "right" in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["right"].append(key_right)
                    else:
                        equivalent_key_group[key_group]["right"] = [key_right]
                else:
                    assert key2.split(".")[0] == left_table, f"unrecognized table alias"
                    key_left = left_table + "." + key2.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["left"].append(key_left)
                    else:
                        equivalent_key_group[key_group]["left"] = [key_left]
                    right_table = key1.split(".")[0]
                    key_right = right_table + "." + key1.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if "right" in equivalent_key_group[key_group]:
                        equivalent_key_group[key_group]["right"].append(key_right)
                    else:
                        equivalent_key_group[key_group]["right"] = [key_right]

            for group in union_key_group_set:
                if group in equivalent_key_group:
                    new_left_key = []
                    for key in table_key_equivalent_group[left_table][group]:
                        if key not in equivalent_key_group[group]["left"]:
                            new_left_key.append(key)
                    if len(new_left_key) != 0:
                        union_key_group[group] = [("left", new_left_key)]
                    new_right_key = []
                    for key in right_bound_factor.table_key_equivalent_group[group]:
                        if key not in equivalent_key_group[group]["right"]:
                            new_right_key.append(key)
                    if len(new_right_key) != 0:
                        if group in union_key_group:
                            union_key_group[group].append(("right", new_right_key))
                        else:
                            union_key_group[group] = [("right", new_right_key)]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                    if group in right_bound_factor.table_key_equivalent_group:
                        if group in union_key_group:
                            union_key_group[group].append(
                                ("right", right_bound_factor.table_key_equivalent_group[group]))
                        else:
                            union_key_group[group] = [("right", right_bound_factor.table_key_equivalent_group[group])]

        else:
            common_key_group = table_equivalent_group[left_table].intersection(right_bound_factor.equivalent_groups)

            if len(common_key_group) > 0:
                assert len(common_key_group) == 1
            else:
                common_key_group = []
            for group in union_key_group_set:
                if group in common_key_group:
                    equivalent_key_group[group] = dict()
                    equivalent_key_group[group]["left"] = table_key_equivalent_group[left_table][group]
                    equivalent_key_group[group]["right"] = right_bound_factor.table_key_equivalent_group[group]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append(("left", table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [("left", table_key_equivalent_group[left_table][group])]
                    if group in right_bound_factor.table_key_equivalent_group:
                        if group in union_key_group:
                            union_key_group[group].append(
                                ("right", right_bound_factor.table_key_equivalent_group[group]))
                        else:
                            union_key_group[group] = [("right", right_bound_factor.table_key_equivalent_group[group])]

        return equivalent_key_group, union_key_group_set, union_key_group, new_join_cond


    def join_two_tables_nc(self, sub_plan_query_str, left_table, right_table, tables_all, conditional_factors, join_keys,
                           table_equivalent_group, table_key_equivalent_group, table_key_group_map, join_cond,
                           ):
        equivalent_key_group, union_key_group_set, union_key_group, new_join_cond = \
            self.get_join_keys_two_tables(left_table, right_table, table_equivalent_group, table_key_equivalent_group,
                                          table_key_group_map, join_cond, join_keys, tables_all)
        cond_factor_left = conditional_factors[left_table]
        cond_factor_right = conditional_factors[right_table]

        new_union_key_group = dict()

        res = cond_factor_left.table_len * cond_factor_right.table_len * cond_factor_left.densities * cond_factor_right.densities

        PKs_in_order = sorted(list(equivalent_key_group.keys()), key=lambda x: self.query_dfs_PKs.index(x))

        prev_left_idx = -1
        prev_right_idx = -1

        for PK in PKs_in_order:
            left_keys = equivalent_key_group[PK][left_table]
            right_keys = equivalent_key_group[PK][right_table]
            assert len(left_keys) == 1, f'left = {left_table}, left keys = {equivalent_key_group[PK][left_table]}, right keys = {equivalent_key_group[PK][right_table]}'
            assert len(right_keys) == 1, f'left = {left_table}, left keys = {equivalent_key_group[PK][left_table]}, right keys = {equivalent_key_group[PK][right_table]}'

            new_union_key_group[PK] = [PK]

            left_idxs = cond_factor_left.PK_to_idx[PK]
            right_idxs = cond_factor_right.PK_to_idx[PK]

            left_col = left_keys[0].split('.')[1]
            right_col = right_keys[0].split('.')[1]

            assert len(left_idxs) == len(right_idxs) == self.num_fact_cols[PK]

            for i, (left_idx, right_idx) in enumerate(zip(left_idxs, right_idxs)):
                assert prev_left_idx < left_idx
                assert prev_right_idx < right_idx

                if i != 0:
                    res = res / torch.maximum(cond_factor_left.min_count[PK], cond_factor_right.min_count[PK])
                    continue
                res = cond_factor_left.P[left_idx] * cond_factor_right.P[right_idx] * self.inv_P[PK][i] * res

        res = torch.mean(res).item()

        for group in union_key_group:
            if group not in new_union_key_group:
                new_union_key_group[group] = []
            for table, keys in union_key_group[group]:
                for key in keys:
                    new_union_key_group[group].append(key)

        new_factor = NCFactor_Group(res)
        new_factor.join_cond = new_join_cond
        new_factor.equivalent_groups = union_key_group_set
        new_factor.table_key_equivalent_group = new_union_key_group

        return new_factor, res

    def get_join_keys_two_tables(self, left_table, right_table, table_equivalent_group, table_key_equivalent_group,
                                 table_key_group_map, join_cond, join_keys, tables_all):
        """
            Get the join keys between two tables
        """
        actual_join_cond = []
        for cond in join_cond[left_table]:
            if cond in join_cond[right_table]:
                actual_join_cond.append(cond)
        equivalent_key_group = dict()
        union_key_group_set = table_equivalent_group[left_table].union(table_equivalent_group[right_table])
        union_key_group = dict()
        new_join_cond = join_cond[left_table].union(join_cond[right_table])
        if len(actual_join_cond) != 0:
            for cond in actual_join_cond:
                key1 = cond.split("=")[0].strip()
                key2 = cond.split("=")[1].strip()
                if key1.split(".")[0] == left_table:
                    key_left = left_table + "." + key1.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        if key_left not in equivalent_key_group[key_group][left_table]:
                            equivalent_key_group[key_group][left_table].append(key_left)
                    else:
                        equivalent_key_group[key_group][left_table] = [key_left]
                    assert key2.split(".")[0] == right_table, f"unrecognized table alias"
                    key_right = right_table + "." + key2.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if right_table in equivalent_key_group[key_group]:
                        if key_right not in equivalent_key_group[key_group][right_table]:
                            equivalent_key_group[key_group][right_table].append(key_right)
                    else:
                        equivalent_key_group[key_group][right_table] = [key_right]
                else:
                    assert key2.split(".")[0] == left_table, f"unrecognized table alias"
                    key_left = left_table + "." + key2.split(".")[-1]
                    key_group = table_key_group_map[left_table][key_left]
                    if key_group not in equivalent_key_group:
                        equivalent_key_group[key_group] = dict()
                    if left_table in equivalent_key_group[key_group]:
                        if key_left not in equivalent_key_group[key_group][left_table]:
                            equivalent_key_group[key_group][left_table].append(key_left)
                    else:
                        equivalent_key_group[key_group][left_table] = [key_left]
                    assert key1.split(".")[0] == right_table, f"unrecognized table alias"
                    key_right = right_table + "." + key1.split(".")[-1]
                    key_group_t = table_key_group_map[right_table][key_right]
                    assert key_group_t == key_group, f"key group mismatch for join {cond}"
                    if right_table in equivalent_key_group[key_group]:
                        if key_right not in equivalent_key_group[key_group][right_table]:
                            equivalent_key_group[key_group][right_table].append(key_right)
                    else:
                        equivalent_key_group[key_group][right_table] = [key_right]

            for group in union_key_group_set:
                if group in equivalent_key_group:
                    new_left_key = []
                    for key in table_key_equivalent_group[left_table][group]:
                        if key not in equivalent_key_group[group][left_table]:
                            new_left_key.append(key)
                    if len(new_left_key) != 0:
                        union_key_group[group] = [(left_table, new_left_key)]
                    new_right_key = []
                    for key in table_key_equivalent_group[right_table][group]:
                        if key not in equivalent_key_group[group][right_table]:
                            new_right_key.append(key)
                    if len(new_right_key) != 0:
                        if group in union_key_group:
                            union_key_group[group].append((right_table, new_right_key))
                        else:
                            union_key_group[group] = [(right_table, new_right_key)]
                else:
                    if group in table_key_equivalent_group[left_table]:
                        if group in union_key_group:
                            union_key_group[group].append((left_table, table_key_equivalent_group[left_table][group]))
                        else:
                            union_key_group[group] = [(left_table, table_key_equivalent_group[left_table][group])]
                    if group in table_key_equivalent_group[right_table]:
                        if group in union_key_group:
                            union_key_group[group].append((right_table, table_key_equivalent_group[right_table][group]))
                        else:
                            union_key_group[group] = [(right_table, table_key_equivalent_group[right_table][group])]

        else:
            common_key_group = table_equivalent_group[left_table].intersection(table_equivalent_group[right_table])
            if len(common_key_group) > 0:
                assert len(common_key_group) == 1
            else:
                common_key_group = []
            for group in union_key_group_set:
                if group in common_key_group:
                    equivalent_key_group[group] = dict()
                    equivalent_key_group[group][left_table] = table_key_equivalent_group[left_table][group]
                    equivalent_key_group[group][right_table] = table_key_equivalent_group[right_table][group]
                elif group in table_key_equivalent_group[left_table]:
                    if group in union_key_group:
                        union_key_group[group].append((left_table, table_key_equivalent_group[left_table][group]))
                    else:
                        union_key_group[group] = [(left_table, table_key_equivalent_group[left_table][group])]
                else:
                    if group in union_key_group:
                        union_key_group[group].append((right_table, table_key_equivalent_group[right_table][group]))
                    else:
                        union_key_group[group] = [(right_table, table_key_equivalent_group[right_table][group])]

        return equivalent_key_group, union_key_group_set, union_key_group, new_join_cond
