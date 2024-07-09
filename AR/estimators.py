"""A suite of cardinality estimators.

In parcticular,
  + ProgressiveSampling: inference algorithms for autoregressive density
    estimators; fanout scaling; draws tuples from trained models subject to
    filters.
  + FactorizedProgressiveSampling: subclass that additionally handles column
    factorization.
"""
import collections
import datetime
import functools
import json
import random
import re
import shutil
import time
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch

import common
import datasets
import distributions
import join_utils
import made
import utils

import copy
import torch.distributions as D
import torch.nn.functional as F

import ctypes
import multiprocessing

VERBOSE = 'VERBOSE' in os.environ

debug = False
debug_get_valids = False

use_ctypes = True
lib = None
use_gpu_valid = True
no_repeat_large_v_list = True

def profile(func):
    return func

def operator_isnull(xs, _unused_val, negate=False):
    ret = pd.isnull(xs)
    return ~ret if negate else ret

def operator_like_ctypes(N, xs, shared_array, like_pattern, negate=False):
    pattern = like_pattern.replace('%', '.*').replace('(', '\(').replace(')', '\)')

    shared_pointer = ctypes.cast(shared_array.get_obj(), ctypes.POINTER(ctypes.c_bool))
    lib.pattern_match(N, xs, ctypes.c_char_p(pattern.encode()), shared_pointer)
    ret = np.array(shared_array[:])

    return ~ret if negate else ret

def operator_true(xs, v):
    return np.ones_like(xs, dtype=np.bool)

def operator_false(xs, v):
    return np.zeros_like(xs, dtype=np.bool)

def operator_notin(xs, v):
    return np.isin(xs,v,invert=True)

def operator_skip(xs, v):
    assert isinstance(v, np.ndarray)
    return v

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '!=': np.not_equal,
    'IN': np.isin,
    'NOT_IN': operator_notin,
    'LIKE': operator_like_ctypes if use_ctypes else operator_like,
    'NOT_LIKE': functools.partial(operator_like_ctypes if use_ctypes else operator_like, negate=True),
    'IS_NULL': operator_isnull,
    'IS_NOT_NULL': functools.partial(operator_isnull, negate=True),
    'ALL_TRUE': operator_true,
    'ALL_FALSE': operator_false,
    'INDEX': operator_skip
}

TOPS = {
    '>': torch.gt,
    '<': torch.lt,
    '!=': torch.ne
}

def GetTablesInQuery(columns, values):
    """Returns a set of table names that are joined in a query."""
    q_tables = set()
    for col, val in zip(columns, values):
        if col.name.startswith('__in_') and val == [1]:
            q_tables.add(col.name[len('__in_'):])
    return q_tables


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.col_prep_starts = 0
        self.col_prep_ms = 0
        self.query_prep_ms = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self,flag=0):
        if flag == 0:
            self.query_starts.append(time.time())
        elif flag == 1:
            self.col_prep_ms = 0
        elif flag == 2:
            self.col_prep_starts = time.time()

    def OnEnd(self,flag=0):
        if flag == 0:
            self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)
        elif flag == 1:
            self.query_prep_ms.append(self.col_prep_ms)
        elif flag == 2:
            self.col_prep_ms += ((time.time() - self.col_prep_starts)*1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards,self.query_prep_ms
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])
        self.query_prep_ms.append(state[5])

    def report(self):
        est = self
        if est.name == 'CardEst':
            est.name = str(est)
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)

        if os[idx] is None:
            os[idx] = [o]
            vs[idx] = [v]
        else:
            # Multiple clauses on same attribute.
            os[idx].append(o)
            vs[idx].append(v)

    return cs, os, vs

def FillInUnqueriedColumnsTree(table, query):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """

    ncols = len(table.columns)
    cs = table.columns
    #os, vs = [None] * ncols, [None] * ncols
    ss = [None] * ncols

    for col in query:
        if '.' in col:
            c = col.split('.')[-1]
            idx = table.ColumnIndex(c)
        else:
            idx = table.ColumnIndex(col)
        ss[idx] = query[col]

    return cs, ss


def ConvertLikeToIn(fact_table, columns, operators, vals):
    assert False
    """Pre-processes a query by converting LIKE predicates to IN predicates.

    Columns refers to the original columns of the table.
    """
    fact_columns = fact_table.columns
    fact_column_names = [fc.name for fc in fact_columns]
    assert len(columns) == len(operators) == len(vals)
    for i in range(len(columns)):
        col, op, val = columns[i], operators[i], vals[i]
        # We don't convert if this column isn't factorized.
        # If original column name isn't in the factorized col names,
        # then this column is factorized.
        # This seems sorta hacky though.
        if op is not None and col.name not in fact_column_names:
            assert len(op) == len(val)
            for j in range(len(op)):
                o, v = op[j], val[j]
                if 'LIKE' in o:
                    new_o = 'IN'
                    valid = OPS[o](col.all_distinct_values, v)
                    new_val = tuple(col.all_distinct_values[valid])
                    op[j] = new_o
                    val[j] = new_val

    assert len(columns) == len(operators) == len(vals)
    return columns, operators, vals


def ConvertLikeToInTree(fact_table, columns, trees, use_raw_table=False):
    fact_columns = fact_table.columns
    fact_column_names = [fc.name for fc in fact_columns]

    operators = [None] * len(columns)
    vals = [None] * len(columns)

    assert len(columns) == len(trees)
    for i in range(len(columns)):
        col, tree = columns[i], trees[i]
        if tree is not None:
            if debug:
                print('converting original tree')
                tree_dict = tree.to_dict()
                print(json.dumps(tree_dict, indent=4))

            def convert_recur(node):
                if 'LIKE' in node.op:
                    data = col.data if use_raw_table else col.all_distinct_values
                    assert data is not None
                    if use_ctypes:
                        real_value = OPS[node.op](len(data),
                                                  col.all_distinct_values_bytes,
                                                  col.shared_array,
                                                  node.value)
                    else:
                        real_value = OPS[node.op](data,
                                                  node.value)
                    node.value = real_value
                    if debug:
                        print('op', OPS[node.op])
                        print('all distinct values or data')
                        print(data)
                        print('valid')
                        print(node.value)
                    node.op = 'LIKE'
                for child in node.children:
                    convert_recur(child)


            convert_recur(tree)

            def merge_or_recur(node, convert_op=False):
                if node.op == "AND":
                    for child in node.children:
                        merge_or_recur(child, convert_op)
                elif node.op == "OR":
                    new_v = None
                    for child in node.children:
                        temp = merge_or_recur(child, True)
                        if new_v is None:
                            new_v = temp
                        else:
                            if isinstance(temp, set):
                                new_v = new_v.union(temp)
                            elif isinstance(temp, np.ndarray):
                                new_v |= temp
                    node.children = []
                    if isinstance(new_v, set):
                        node.op = "IN"
                        node.value = list(new_v)
                    else:
                        node.op = "LIKE"
                        node.value = new_v
                    return new_v
                elif convert_op:
                    if node.op == 'IN':
                        return set(node.value)
                    elif node.op == '=':
                        return set([node.value])
                    elif node.op == 'LIKE':
                        return node.value
                    else:
                        assert False

            merge_or_recur(tree)

            def check_no_or(node):
                assert node.op != "OR"
                for child in node.children:
                    check_no_or(child)

            check_no_or(tree)

            operators[i] = []
            vals[i] = []
            def add_recur(node):
                if node.op not in ["AND", "OR"]:
                    operators[i].append(node.op)
                    if node.op == "IN":
                        assert isinstance(node.value, list) or isinstance(node.value, tuple), f'node.value type {type(node.value)} {node.value}'
                        vals[i].append(node.value)
                    else:
                        vals[i].append(node.value)
                for child in node.children:
                    add_recur(child)

            add_recur(tree)

            def intersect(l1, l2):
                return [v for v in l1 if v in l2]


            in_val = None
            like_val = None
            new_operators = []
            new_vals = []

            for op, val in zip(operators[i], vals[i]):
                if op == "IN":
                    if in_val is None:
                        in_val = val
                    else:
                        if len(in_val) < len(val):
                            in_val = intersect(in_val, val)
                        else:
                            in_val = intersect(val, in_val)
                elif op == "LIKE":
                    if like_val is None:
                        like_val = val
                    else:
                        like_val &= val
                else:
                    new_operators.append(op)
                    new_vals.append(val)

            if in_val is not None:
                new_operators.append("IN")
                new_vals.append(in_val)
            if like_val is not None:
                new_operators.append("INDEX")
                new_vals.append(like_val)

            assert "LIKE" not in new_operators
            assert "NOT_LIKE" not in new_operators
            assert "NOT LIKE" not in new_operators
            operators[i] = new_operators
            vals[i] = new_vals

    return columns, operators, vals


def ProjectQuery(fact_table, columns, operators, vals, trees=None, use_raw_table=False):
    """Projects these cols, ops, and vals to factorized version.

    Returns cols, ops, vals, dominant_ops where all lists are size
    len(fact_table.columns), in the factorized table's natural column order.

    Dominant_ops is None for all operators that don't use masking.
    <, >, <=, >=, IN all use masking.
    """
    if trees is None:
        assert False
        columns, operators, vals = ConvertLikeToIn(fact_table, columns, operators,
                                                   vals)
    else:
        columns, operators, vals = ConvertLikeToInTree(fact_table, columns, trees, use_raw_table)
        if debug:
            for col, op, val in zip(columns, operators, vals):
                print('after ConvertLikeToInTree col', col, 'op', op, 'val', val)

    nfact_cols = len(fact_table.columns)
    fact_cols = []

    fact_ops = [None] * nfact_cols
    fact_vals = [None] * nfact_cols
    fact_dominant_ops = [None] * nfact_cols

    # i is the index of the current factorized column.
    # j is the index of the actual table column that col_i corresponds to.
    j = -1
    for i in range(nfact_cols):
        col = fact_table.columns[i]
        if col.factor_id in [None, 0]:
            j += 1
        op = operators[j]
        fact_cols.append(col)

        if op is not None:
            val = vals[j]
            if fact_ops[i] is None:
                fact_ops[i] = []
                fact_vals[i] = []
                fact_dominant_ops[i] = []
            val_list = None
            for o, v in zip(op, val):
                if col.factor_id is None:
                    fact_ops[i].append(o)
                    fact_vals[i].append(v)
                    fact_dominant_ops[i] = None
                else:
                    if o == 'IN' or o == 'INDEX':
                        if len(v) == 0:
                            fact_ops[i].append('ALL_FALSE')
                            fact_vals[i].append(None)
                            fact_dominant_ops[i].append(None)
                        else:
                            if val_list is None:
                                if o == "INDEX":
                                    v = v.nonzero()[0]
                                    assert isinstance(v, np.ndarray), f'v type {type(v)} value {v}'
                                    val_list = v
                                else:
                                    if type(v) == str:
                                        v = (v,)
                                    val_list = np.array(list(v))
                                    val_list = common.Discretize(
                                        columns[j], True, val_list, fail_out_of_domain=False,
                                                                 use_val_to_bin=True)
                            assert len(val_list) <= len(columns[j].all_distinct_values)
                            if len(val_list) == 0:
                                fact_ops[i].append('ALL_FALSE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            elif len(val_list) != len(columns[j].all_distinct_values):
                                if use_gpu_valid and no_repeat_large_v_list:
                                    if col.factor_id == 0:
                                        mask_shape = []
                                        mask_loc = []
                                        p_v_list = np.vectorize(col.ProjectValue)(val_list)
                                        mask_shape.append(len(col.all_distinct_values))
                                        mask_loc.append(p_v_list)
                                        k = i + 1
                                        while True:
                                            next_col = fact_table.columns[k]
                                            if next_col.factor_id in [None, 0]:
                                                break
                                            next_p_v_list = np.vectorize(next_col.ProjectValue)(val_list)
                                            mask_shape.append(len(next_col.all_distinct_values))
                                            mask_loc.append(next_p_v_list)
                                            k = k + 1
                                        assert len(mask_shape) == 2
                                        assert len(mask_loc) == 2
                                        mask = torch.zeros(*mask_shape, dtype=np.bool).cuda()
                                        assert mask.shape == tuple(mask_shape), f'mask shape {mask.shape} intended {mask_shape}'
                                        mask[mask_loc[0], mask_loc[1]] = True

                                        fact_vals[i].insert(0, mask.max(1).values)
                                        assert fact_vals[i][0].shape == (mask_shape[0],)
                                        fact_ops[i].insert(0, 'IN')
                                        fact_dominant_ops[i].insert(0, "IN")

                                        if fact_ops[i+1] is None:
                                            assert fact_vals[i+1] is None
                                            assert fact_dominant_ops[i+1] is None
                                            fact_ops[i+1] = []
                                            fact_vals[i+1] = []
                                            fact_dominant_ops[i+1] = []
                                        fact_vals[i+1].insert(0, mask)
                                        fact_ops[i+1].insert(0, 'IN')
                                        fact_dominant_ops[i+1].insert(0, "IN")
                                else:
                                    assert False
                            else:
                                fact_ops[i].append('ALL_TRUE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                    elif o == 'NOT_IN':
                        if len(v) == 0:
                            fact_ops[i].append('ALL_TRUE')
                            fact_vals[i].append(None)
                            fact_dominant_ops[i].append(None)
                        else:
                            if val_list is None:
                                if type(v) == str:
                                    v = (v,)
                                val_list = np.array(list(v))
                                val_list = common.Discretize(
                                    columns[j], True, val_list, fail_out_of_domain=False,
                                                             use_val_to_bin=True)
                            assert len(val_list) <= len(columns[j].all_distinct_values)
                            if len(val_list) == 0:
                                fact_ops[i].append('ALL_TRUE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            elif len(val_list) != len(columns[j].all_distinct_values):
                                if use_gpu_valid and no_repeat_large_v_list:
                                    if col.factor_id == 0:
                                        mask_shape = []
                                        mask_loc = []
                                        p_v_list = np.vectorize(col.ProjectValue)(val_list)
                                        mask_shape.append(len(col.all_distinct_values))
                                        mask_loc.append(p_v_list)

                                        k = i + 1
                                        while True:
                                            next_col = fact_table.columns[k]
                                            if next_col.factor_id in [None, 0]:
                                                break
                                            next_p_v_list = np.vectorize(next_col.ProjectValue)(val_list)
                                            mask_shape.append(len(next_col.all_distinct_values))
                                            mask_loc.append(next_p_v_list)
                                            k = k + 1
                                        assert len(mask_shape) == 2
                                        assert len(mask_loc) == 2
                                        mask = torch.ones(*mask_shape, dtype=np.bool).cuda()
                                        assert mask.shape == tuple(mask_shape), f'mask shape {mask.shape} intended {mask_shape}'
                                        mask[mask_loc[0], mask_loc[1]] = False

                                        fact_vals[i].insert(0, mask.max(1).values)
                                        assert fact_vals[i][0].shape == (mask_shape[0],)
                                        fact_ops[i].insert(0, 'IN')
                                        fact_dominant_ops[i].insert(0, "IN")

                                        if fact_ops[i+1] is None:
                                            assert fact_vals[i+1] is None
                                            assert fact_dominant_ops[i+1] is None
                                            fact_ops[i+1] = []
                                            fact_vals[i+1] = []
                                            fact_dominant_ops[i+1] = []
                                        fact_vals[i+1].insert(0, mask)
                                        fact_ops[i+1].insert(0, 'IN')
                                        fact_dominant_ops[i+1].insert(0, "IN")
                            else:
                                fact_ops[i].append('ALL_FALSE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)

                    elif 'NULL' in o:
                        assert columns[j].hasnan is not None
                        if columns[j].hasnan:
                            if o == 'IS_NULL':
                                fact_ops[i].append(col.ProjectOperator('='))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(None)
                            elif o == 'IS_NOT_NULL':
                                fact_ops[i].append(col.ProjectOperator('>'))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(
                                    col.ProjectOperatorDominant('>'))
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                        else:
                            if o == 'IS_NULL':
                                new_op = 'ALL_FALSE'
                            elif o == 'IS_NOT_NULL':
                                new_op = 'ALL_TRUE'
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                            fact_ops[i].append(new_op)
                            fact_vals[i].append(None)
                            fact_dominant_ops[i].append(None)
                    else:
                        assert columns[j].hasnan is not None
                        if o in ['<=', '<', '!='] and columns[j].hasnan:
                            fact_ops[i].append(col.ProjectOperator('>'))
                            fact_vals[i].append(col.ProjectValue(0))
                            fact_dominant_ops[i].append(
                                col.ProjectOperatorDominant('>'))
                        if v not in columns[j].all_distinct_values:
                            if o == '=':
                                fact_ops[i].append('ALL_FALSE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            elif o == '!=':
                                fact_ops[i].append('ALL_TRUE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            else:
                                f = columns[j].all_distinct_values[0]
                                if type(v) == str and type(f) == float and np.isnan(f):
                                    columns[j].all_distinct_values[0] = str('')
                                if o == '<' or o == '>=':
                                    value = np.nonzero(columns[j].all_distinct_values >= v)[0][0]
                                elif o == '>' or o == '<=':
                                    value = np.nonzero(columns[j].all_distinct_values <= v)[0][-1]
                                p_v = col.ProjectValue(value)
                                p_op = col.ProjectOperator(o)
                                p_dom_op = col.ProjectOperatorDominant(o)
                                fact_ops[i].append(p_op)
                                fact_vals[i].append(p_v)
                                if p_dom_op in common.PROJECT_OPERATORS_DOMINANT.values():
                                    fact_dominant_ops[i].append(p_dom_op)
                                else:
                                    fact_dominant_ops[i].append(None)
                        else:
                            value = np.nonzero(
                                columns[j].all_distinct_values == v)[0][0]
                            p_v = col.ProjectValue(value)
                            p_op = col.ProjectOperator(o)
                            p_dom_op = col.ProjectOperatorDominant(o)
                            fact_ops[i].append(p_op)
                            fact_vals[i].append(p_v)
                            if p_dom_op in common.PROJECT_OPERATORS_DOMINANT.values():
                                fact_dominant_ops[i].append(p_dom_op)
                            else:
                                fact_dominant_ops[i].append(None)
                    assert fact_ops[-1] != "INDEX"
                    assert fact_dominant_ops[-1] != "INDEX"

    assert len(fact_cols) == len(fact_ops) == len(fact_vals) == len(
        fact_dominant_ops) == nfact_cols

    return fact_cols, fact_ops, fact_vals, fact_dominant_ops


def _infer_table_names(columns):
    ret = []
    for col in columns:
        if col.name.startswith('__in_') and col.factor_id in [None, 0]:
            ret.append(col.name[len('__in_'):])
    return ret


class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            join_spec=None,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False,  # Skip sampling on wildcards?
            do_fanout_scaling=False,
    ):
        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.model.eval()
        self.table = table
        self.all_tables = set(join_spec.join_tables
                             ) if join_spec is not None else _infer_table_names(
                                 table.columns)
        self.join_graph = join_spec.join_graph
        self.shortcircuit = shortcircuit
        self.do_fanout_scaling = do_fanout_scaling
        if do_fanout_scaling:
            self.num_tables = len(self.all_tables)
            print('# tables in join schema:', self.num_tables)

            self.primary_table_name = join_spec.join_root

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        print('set cardinality', cardinality)
        if cardinality is None:
            self.cardinality = table.cardinality
            print('set table cardinality', table.cardinality)

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        ########### Inference optimizations below.
        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            self.inp = self.inp.view(self.num_samples, -1)


    def set_verbose_data(self,num_query):
        if 'VERBOSE' in os.environ:
            self.query_info = dict()
            for i in range(num_query) : self.query_info[i] = dict()
            self.cur_query = -1

    # +@ FOR VERBOSE FILE
    def save_verbose_data(self,workload,sep='\t'):
        now = datetime.datetime.now().strftime('%m%d')
        path = f"./results/{workload}/{workload}_{str(self)}_inference_[{now}].tsv"
        with open(path,'wt') as file :
            # write header
            header_list = self.query_info[0].keys()
            header = f"query_num{sep}{sep.join(header_list)}"
            file.write(header+'\n')
            # write rows
            for k,v in self.query_info.items() :
                values = v.values()
                values = list(map(lambda x: str(x).replace('\t',''), values))
                value_txt = f"{sep.join(values)}"
                line = f"{k}{sep}{value_txt}\n"
                file.write(line)


    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _maybe_remove_nan(self, dvs):
        # NOTE: "dvs[0] is np.nan" or "dvs[0] == np.nan" don't work.
        #if dvs.dtype == np.dtype('object') and pd.isnull(dvs[0]):
        if pd.isnull(dvs[0]):
            return dvs[1:], True
        return dvs, False

    def _truncate_val_string(self, val):
        truncated_vals = []
        for v in val:
            if type(v) == tuple:
                new_val = str(list(v)[:20]) + '...' + str(
                    len(v) - 20) + ' more' if len(v) > 20 else list(v)
            else:
                new_val = v
            truncated_vals.append(new_val)
        return truncated_vals

    def _print_probs(self, columns, operators, vals, ordering, masked_logits):
        ml_i = 0
        txt = ''
        for i in range(len(columns)):
            natural_idx = ordering[i]
            if operators[natural_idx] is None:
                continue
            truncated_vals = self._truncate_val_string(vals[natural_idx])
            txt += f"P({columns[natural_idx].name} {operators[natural_idx]} {truncated_vals} | past) ~= {masked_logits[ml_i].mean().cpu().item():.6f}, "
            ml_i += 1
        col_probs = txt
        self.query_info[self.cur_query]['col_probs'] = col_probs



    @profile
    def get_probs_for_col(self, logits, natural_idx, num_classes):
        """Returns probabilities for column i given model and logits."""
        num_samples = logits.size()[0]
        if self.model.UseDMoL(natural_idx):
            dmol_params = self.model.logits_for_col(
                natural_idx, logits)
            logits_i = torch.zeros((num_samples, num_classes),
                                   device=self.device)
            for i in range(num_classes):
                logits_i[:, i] = distributions.dmol_query(
                    dmol_params,
                    torch.ones(num_samples, device=self.device) * i,
                    num_classes=num_classes,
                    num_mixtures=self.model.num_dmol,
                    scale_input=self.model.scale_input)
        else:
            logits_i = self.model.logits_for_col(
                natural_idx, logits)

        return torch.softmax(logits_i, 1)

    def _get_fanout_columns(self, cols, vals):
        # What tables are joined in this query?
        q_tables = GetTablesInQuery(cols, vals)
        some_query_table = next(iter(q_tables))  # pick any table in the query
        fanout_tables = self.all_tables - q_tables

        # For each table not in the query, find a path to q_table. The first
        # edge in the path gives the correct fanout column. We use
        # `shortest_path` here but the path is actually unique.
        def get_fanout_key(u):
            if self.join_graph is None:
                return None
            path = nx.shortest_path(self.join_graph, u, some_query_table)
            v = path[1]  # the first hop from the starting node u
            join_key = self.join_graph[u][v]["join_keys"][u]
            return join_key

        fanout_cols = [(t, get_fanout_key(t)) for t in fanout_tables]

        # The fanouts from a "primary" table are always 1.
        return list(
            filter(lambda tup: tup[0] != self.primary_table_name, fanout_cols))

    def _get_fanout_column_index(self, fanout_col):
        """Returns the natural index of a fanout column.

        For backward-compatibility, try both `__fanout_{table}` and
        `__fanout_{table}__{col}`.
        """
        table, key = fanout_col
        for col_name in [
                '__fanout_{}__{}'.format(table, key),
                '__fanout_{}'.format(table)
        ]:
            if col_name in self.table.name_to_index:
                return self.table.ColumnIndex(col_name)
        assert False, (fanout_col, self.table.name_to_index)

    @profile
    def _scale_probs(self, columns, operators, vals, p, ordering, num_fanouts,
                     num_indicators, inp):

        # Find out what foreign tables are not present in this query.
        fanout_cols = self._get_fanout_columns(columns, vals)
        indexes_to_scale = [
            self._get_fanout_column_index(col) for col in fanout_cols
        ]

        if len(indexes_to_scale) == 0:
            return p.mean().item()

        # Make indexes_to_scale conform to sampling ordering.  No-op if
        # natural ordering is used.
        if isinstance(ordering, np.ndarray):
            ordering = list(ordering)
        sampling_order_for_scale = [
            ordering.index(natural_index) for natural_index in indexes_to_scale
        ]
        zipped = list(zip(sampling_order_for_scale, indexes_to_scale))
        zipped = sorted(zipped, key=lambda t: t[0])
        sorted_indexes_to_scale = list(map(lambda t: t[1], zipped))

        scale = 1.0

        for natural_index in sorted_indexes_to_scale:
            # Sample the fanout factors & feed them back as input.  This
            # modeling of AR dependencies among the fanouts improves errors
            # (than if they were just dependent on the content+indicators).
            logits = self._forward_encoded_input(inp,
                                                 sampling_ordering=ordering)

            # The fanouts are deterministic function based on join key.  We
            # can either model the join keys, and thus can sample a join
            # key and lookup.  Alternatively, we directly model the fanouts
            # and sample them.  Sample works better than argmax or
            # expectation.
            fanout_probs = self.get_probs_for_col(
                logits, natural_index, columns[natural_index].distribution_size)

            # Turn this on when measuring inference latency: multinomial() is
            # slightly faster than Categorical() then sample().  The latter has
            # a convenient method for printing perplexity though.
            # scales = torch.multinomial(fanout_probs, 1)
            dist = torch.distributions.categorical.Categorical(fanout_probs)
            scales = dist.sample()

            # Off-by-1 in fanout's domain: 0 -> np.nan, 1 -> value 0, 2 ->
            # value 1.
            actual_scale_values = (scales - 1).clamp_(1)

            if 'VERBOSE' in os.environ:

                # print('scaling', columns[natural_index], 'with',
                #       actual_scale_values.float().mean().item(), '; perplex.',
                #       dist.perplexity()[:3])
                scaling = f"{columns[natural_index]} with {actual_scale_values.float().mean().item()}"
                perplex = dist.perplexity()[:3]
                self.query_info[self.cur_query]['scaling'] = scaling
                self.query_info[self.cur_query]['perplex'] = perplex





            scale *= actual_scale_values

            # Put the sampled 'scales' back into input, so that fanouts to
            # be sampled next can depend on the current fanout value.
            self._put_samples_as_input(scales.view(-1, 1), inp, natural_index)
        if 'VERBOSE' in os.environ:
        # if os.environ.get('VERBOSE', None) == '2':
            # print('  p quantiles',
            #       np.quantile(p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]), 'mean',
            #       p.mean())
            # print('  scale quantiles',
            #       np.quantile(scale.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
            #       'mean', scale.mean())

            # +@ add modi
            p_quantile = np.quantile(p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0])
            p_quantile_mean = p.mean()
            scale_quantiles = np.quantile(scale.cpu().numpy(), [0.5, 0.9, 0.99, 1.0])
            scale_quantiles_mean  = scale.mean()

            self.query_info[self.cur_query]['p_quantile'] = p_quantile
            self.query_info[self.cur_query]['p_quantile_mean'] = p_quantile_mean
            self.query_info[self.cur_query]['scale_quantiles'] = scale_quantiles
            self.query_info[self.cur_query]['scale_quantiles_mean'] = scale_quantiles_mean

        scaled_p = p / scale.to(torch.float)

        if 'VERBOSE' in os.environ:
        # if os.environ.get('VERBOSE', None) == '2':
            # print('  scaled_p quantiles',
            #       np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
            #       'mean', scaled_p.mean())
            scaled_p_quantiles =np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0])
            scaled_p_quantiles_mean = scaled_p.mean()
            self.query_info[self.cur_query]['scaled_p_quantiles'] = scaled_p_quantiles
            self.query_info[self.cur_query]['scaled_p_quantiles_mean'] = scaled_p_quantiles_mean


        # NOTE: overflow can happen for undertrained models.
        scaled_p[scaled_p == np.inf] = 0

        # +@ modi condition
        if 'VERBOSE' in os.environ:
        # if os.environ.get('VERBOSE', None) == '2':
            # print('  (after clip) scaled_p quantiles',
            #       np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
            #       'mean', scaled_p.mean())
            after_clip_scaled_p_quantiles =  np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0])
            after_clip_scaled_p_quantiles_mean = scaled_p.mean()
            self.query_info[self.cur_query]['after_clip_scaled_p_quantiles'] = after_clip_scaled_p_quantiles
            self.query_info[self.cur_query]['after_clip_scaled_p_quantiles_mean'] = after_clip_scaled_p_quantiles_mean

        return scaled_p.mean().item()

    @profile
    def _put_samples_as_input(self,
                              data_to_encode,
                              inp,
                              natural_idx,
                              sampling_order_idx=None
                              ):
        """Puts [bs, 1] sampled values approipately into inp."""
        if natural_idx == 0:
            self.model.EncodeInput(
                data_to_encode,
                natural_col=0,
                out=inp[:, :self.model.input_bins_encoded_cumsum[0]]
                )
        else:
            l = self.model.input_bins_encoded_cumsum[natural_idx - 1]
            r = self.model.input_bins_encoded_cumsum[natural_idx]
            self.model.EncodeInput(data_to_encode,
                                   natural_col=natural_idx,
                                   out=inp[:, l:r]
                                   )
    @profile
    def _forward_encoded_input(self, inp, sampling_ordering):
        if debug:
            print(f'(forward_encoded_input) inp {inp} order {sampling_ordering}')
        if hasattr(self.model, 'do_forward'):
            if isinstance(self.model, made.MADE):
                inv = utils.InvertOrder(sampling_ordering)
                logits = self.model.do_forward(inp, inv)
            else:
                logits = self.model.do_forward(inp, sampling_ordering)
        else:
            if self.traced_fwd is not None:
                logits = self.traced_fwd(inp)
            else:
                logits = self.model.forward_with_encoded_input(inp)
        return logits

    def put_sample(self, data_to_encode, sample, natural_idx):
        self._put_samples_as_input(data_to_encode,
                                   sample,
                                   natural_idx,
                                   sampling_order_idx=natural_idx)
        return sample

    @profile
    def _infer_table(self,
                     ordering,
                     columns,
                     operators,
                     vals,
                     stop_keys=None,
                     start_idx=None,
                     return_idx=None,
                     computed_mask=None
                     ):
        p = None
        nrows = None
        ncols = len(columns)
        mask_i_list = [None] * ncols

        col_range = range(ncols) if return_idx is None else range(start_idx, return_idx + 1)
        for i in col_range:
            natural_idx = i if ordering is None else ordering[i]
            assert columns[natural_idx].data is None
            assert columns[natural_idx].factor_id in [None, 0]
            data = self.base_table.columns[natural_idx].data
            nr = len(data)

            if nrows is None:
                nrows = nr
            else:
                assert nrows == nr

            if stop_keys is not None:
                stop = False
                for key in stop_keys:
                    if columns[natural_idx].name == key or columns[natural_idx].name.startswith(key + '_fact_'):
                        stop = True
                        break
                if stop:
                    break

            if (not self.shortcircuit) or (operators[natural_idx] is not None):
                mask, has_mask = self._get_valids(
                    data,
                    operators[natural_idx],
                    vals[natural_idx] if vals else None,
                    natural_idx,
                    1
                )
                assert not has_mask

                assert use_gpu_valid
                mask_i_list[i] = mask
                assert mask.shape == (nrows,), f'nrows {nrows} but mask shape {mask.shape}'

            if return_idx is not None and return_idx == natural_idx:
                num_classes = len(columns[natural_idx].all_distinct_values)

                if mask_i_list[i] is not None:
                    assert data.shape == mask_i_list[i].shape
                    if computed_mask is None:
                        computed_mask = mask_i_list[i]
                    else:
                        assert mask_i_list[i].shape == computed_mask.shape
                        computed_mask *= mask_i_list[i]

                if computed_mask is not None:
                    masked_data = torch.as_tensor(data, device=self.device) * computed_mask
                    masked_nrows = torch.sum(computed_mask)
                else:
                    masked_data = torch.as_tensor(data, device=self.device)
                    masked_nrows = nrows

                scale = 1.0
                if mask_i_list[i] is not None:
                    scale = torch.sum(mask_i_list[i]) / masked_nrows

                p = torch.bincount(masked_data)

                non_masked_nrows = nrows - masked_nrows
                p[0] = p[0] - non_masked_nrows

                p = torch.cat((torch.zeros(1, device=self.device), p), dim=0)

                if p.shape[0] < num_classes:
                    zeros = torch.zeros(num_classes - p.shape[0], device=self.device)
                    p = torch.cat((p, zeros), dim=0)
                assert p.shape[0] == num_classes
                p = p / masked_nrows * scale
                assert torch.isclose(torch.sum(p), torch.ones(1, device=self.device) * scale), f'nrows {nrows} masked nrows {masked_nrows} p {p} p sum {torch.sum(p)} scale {scale}'
                break

            if i == ncols - 1:
                break

        if p is None:
            m = None
            for mask in mask_i_list:
                if mask is not None:
                    if m is None:
                        m = mask
                    else:
                        assert m.shape == mask.shape, f'm shape {m.shape} mask shape {mask.shape}'
                        m = m * mask
            if m is None:
                p = torch.ones(1, device=self.device)
            else:
                p = torch.sum(m) / nrows
        else:
            m = computed_mask

        return p, m, natural_idx


    @profile
    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None,
                  sample_alg='ps',
                  trees=None,
                  return_sample=False,
                  stop_keys=None,
                  start_idx=None,
                  return_idx=None,
                  logits=None,
                  subvar_dropout=False,
                  adjust_fact_col=False
                  ):

        torch.set_grad_enabled(False)
        ncols = len(columns)
        if logits is None:
            logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []
        valid_i_list = [None] * ncols

        self.OnStart(flag=1)
        col_range = range(ncols) if return_idx is None else range(start_idx, return_idx + 1)
        computed_logits = True
        for i in col_range:
            natural_idx = i if ordering is None else ordering[i]
            if stop_keys is not None:
                stop = False
                for key in stop_keys:
                    if columns[natural_idx].name == key or columns[natural_idx].name.startswith(key + '_fact_'):
                        stop = True
                        break
                if stop:
                    break
            if i != 0:
                num_i = 1
            else:
                num_i = num_samples if num_samples else int(
                    self.r * self.dom_sizes[natural_idx])

            if self.shortcircuit and operators[natural_idx] is None:
                if return_idx is not None and return_idx == natural_idx:
                    num_classes = len(columns[natural_idx].all_distinct_values)
                    if not computed_logits:
                        computed_logits = True
                        logits = self._forward_encoded_input(inp,
                                                             sampling_ordering=ordering)
                    assert computed_logits
                    probs_i = self.get_probs_for_col(logits,
                                                     natural_idx,
                                                     num_classes=num_classes)
                    break
                self._put_samples_as_input(None, inp, natural_idx)
                data_to_encode = None
                computed_logits = False
            else:
                assert not self.shortcircuit or operators[natural_idx] is not None
                if subvar_dropout and adjust_fact_col and return_idx is not None:
                    if return_idx != natural_idx and len(columns[natural_idx].all_distinct_values) == 1:
                        self._put_samples_as_input(None, inp, natural_idx)
                        data_to_encode = None
                        computed_logits = False
                        continue
                if self.model.output_encoding is not None or self.model.UseDMoL(natural_idx):
                    dvs = columns[natural_idx].all_distinct_values
                    self.OnStart(flag=2)
                    valid_i, has_mask = self._get_valids(
                        dvs, operators[natural_idx], vals[natural_idx] if vals else None,
                        natural_idx, num_samples)
                    self.OnEnd(flag=2)

                    if use_gpu_valid:
                        if not isinstance(valid_i, torch.Tensor):
                            assert valid_i == 1.0
                            valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)
                        else:
                            valid_i_list[i] = valid_i
                    else:
                        valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)
                    num_classes = len(dvs)
                    if not computed_logits:
                        computed_logits = True
                        logits = self._forward_encoded_input(inp,
                                                             sampling_ordering=ordering)
                    probs_i = self.get_probs_for_col(logits,
                                                     natural_idx,
                                                     num_classes=num_classes)

                    valid_i = valid_i_list[i]
                    if valid_i is not None:
                        probs_i *= valid_i
                    probs_i_summed = probs_i.sum(1)

                    if return_idx is not None and return_idx == natural_idx:
                        break
                    masked_probs.append(probs_i_summed)
                    if i == ncols - 1:
                        break

                    paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                    probs_i = probs_i.masked_fill_(paths_vanished, 1.0)

                    samples_i = torch.multinomial(probs_i,
                                                  num_samples=num_i,
                                                  replacement=True)

                    probs_i = probs_i.masked_fill_(paths_vanished, 0.0)
                    if has_mask:
                        self.update_factor_mask(samples_i.view(-1,),
                                                vals[natural_idx] if vals else None, natural_idx)

                    data_to_encode = samples_i.view(-1, 1)

                assert data_to_encode is not None
                assert i == natural_idx
                self._put_samples_as_input(data_to_encode,
                                           inp,
                                           natural_idx,
                                           sampling_order_idx=i)

            if i == ncols - 1:
                break

            next_natural_idx = i + 1 if ordering is None else ordering[i + 1]
            if self.shortcircuit and operators[next_natural_idx] is None:
                continue

            computed_logits = True
            logits = self._forward_encoded_input(inp,
                                                 sampling_ordering=ordering)
        self.OnEnd(flag=1)

        # Debug outputs.
        if 'VERBOSE' in os.environ:
            self._print_probs(columns, operators, vals, ordering, masked_probs)

        if return_idx is not None:
            p = probs_i
            if len(masked_probs) > 0:
                for ls in masked_probs:
                    assert ls.shape == (num_samples,)
                    p *= ls.view(-1,1)
            if p.shape[0] != num_samples:
                p = p.expand(num_samples, -1)
            assert p.shape == (num_samples, num_classes), f'p shape {p.shape} {p} != ({num_samples}, {num_classes})'
            assert len(p.shape) == 2
            assert p.shape[0] == num_samples
        else:
            if len(masked_probs) > 1:
                p = masked_probs[1]
                for ls in masked_probs[2:]:
                    p *= ls
                p *= masked_probs[0]
            elif len(masked_probs) == 1:
                p = masked_probs[0]
            else:
                p = torch.tensor([1.], device=self.device)

        if not computed_logits:
            logits = self._forward_encoded_input(inp,
                                                 sampling_ordering=ordering)
        return p, inp, logits, natural_idx

    def _StandardizeQuery(self, columns, operators, vals):
        return FillInUnqueriedColumns(self.table, columns, operators, vals)

    def QueryTree(self, query: dict, sample_alg='ps', keys=None, sample_size=2048, use_raw_table=False, subvar_dropout=False, adjust_fact_col=False):
        columns, operators, vals, dominant_ops = self._StandardizeQueryTree(query, use_raw_table)

        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        with torch.no_grad():
            # set sample to zero
            inp_buf = self.inp.zero_()

            if num_orderings == 1:
                ordering = orderings[0]
                inv_ordering = utils.InvertOrder(ordering)
                self.OnStart()
                if use_raw_table:
                    p, m, cur_idx = self._infer_table(
                        inv_ordering,
                        columns,
                        operators,
                        vals,
                        stop_keys=keys
                    )
                    self.OnEnd()
                    return p, self.cardinality, cur_idx, None, m, columns, operators, vals, dominant_ops
                else:
                    p, sample, logits, cur_idx = self._sample_n(
                        sample_size,
                        inv_ordering,
                        columns,
                        operators,
                        vals,
                        inp=inp_buf,
                        sample_alg=sample_alg,
                        return_sample=True,
                        stop_keys=keys,
                        subvar_dropout=subvar_dropout,
                        adjust_fact_col=adjust_fact_col
                    )
                self.OnEnd()
                return p, self.cardinality, cur_idx, sample.detach().clone(), logits.detach().clone(), columns, operators, vals, dominant_ops

            assert False


class FactorizedProgressiveSampling(ProgressiveSampling):
    """Additional logic for handling column factorization."""

    def __init__(self,
                 model,
                 fact_table,
                 r,
                 join_spec=None,
                 device=None,
                 seed=False,
                 cardinality=None,
                 shortcircuit=False,
                 do_fanout_scaling=None):
        self.fact_table = fact_table
        self.base_table = fact_table.base_table
        self.factor_mask = None

        super(FactorizedProgressiveSampling,
              self).__init__(model, fact_table, r, join_spec, device, seed,
                             cardinality, shortcircuit, do_fanout_scaling)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.base_table.columns[0].DistributionSize())
        return 'fact_psample_{}'.format(n)

    def get_all_col_names(self):
        return [col.name for col in self.fact_table.columns]

    def get_natural_idx(self, key):
        ret = []

        for i, col in enumerate(self.fact_table.columns):
            if key == col.name or col.name.startswith(key + '_fact_'):
                ret.append(i)
        return ret

        if key:
            ret.append(idx)
        else:
            i = 0
            while True:
                k = key + f'_fact_{i}'
                idx = self.base_table.ColumnIndex(k)
                if idx:
                    ret.append(idx)
                else:
                    break
        assert len(ret) > 0
        return ret

    def get_col_name(self, idx):
        col = self.fact_table.columns[idx]
        return col.name

    def get_domain(self, idx):
        col = self.fact_table.columns[idx]
        if debug:
            print('col', type(col), col.name, col.distribution_size)
        assert len(col.all_distinct_values) == col.distribution_size
        return col.all_distinct_values

    def make_bytes_domain(self, use_raw_table):
        assert use_ctypes

        global lib

        if lib is None:
            libname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "mylib.so"))
            lib = ctypes.CDLL(libname)

        for col in self.base_table.columns:
            data = col.data if use_raw_table else col.all_distinct_values
            assert data is not None
            N = len(data)
            assert N > 0

            is_str = [isinstance(v, str) for v in data if not pd.isnull(v) and v is not None]
            if len(is_str) > 0 and any(is_str):
                xs = [None if pd.isnull(val) or val is None
                      else ctypes.c_char_p(val.encode()) if len(val) < 4096 else ctypes.c_char_p(val[:4096].encode())
                      for val in data]
                assert len(xs) == N
                StringArrayN = ctypes.c_char_p * N
                col.all_distinct_values_bytes = StringArrayN(*xs)
                col.shared_array = multiprocessing.Array(ctypes.c_bool, N)
                assert len(col.all_distinct_values_bytes) == N
                assert len(col.shared_array) == N
            col.hasnan = pd.isnull(data).any()
            col.val_to_bin = {v: i for i, v in enumerate(col.all_distinct_values)}

    def get_P(self, start_idx, return_idx, sample_size, sample, logits, cols, ops, vals, dom_ops, use_raw_table=False, subvar_dropout=False, adjust_fact_col=False):
        self.dominant_ops = dom_ops
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        with torch.no_grad():
            ordering = orderings[0]
            inv_ordering = utils.InvertOrder(ordering)

            if use_raw_table:
                p, l, cur_idx = self._infer_table(
                    inv_ordering,
                    cols,
                    ops,
                    vals,
                    start_idx=start_idx,
                    return_idx=return_idx,
                    computed_mask=logits
                )
                s = None
            else:
                p, s, l, cur_idx = self._sample_n(
                    sample_size,
                    inv_ordering,
                    cols,
                    ops,
                    vals,
                    inp=sample,
                    return_sample=True,
                    start_idx=start_idx,
                    return_idx=return_idx,
                    logits=logits,
                    subvar_dropout=subvar_dropout,
                    adjust_fact_col=adjust_fact_col
                )

        return p, s, l, cur_idx

    def _StandardizeQuery(self, columns, operators, vals):
        assert False
        self.original_query = (columns, operators, vals)
        cols, ops, vals = FillInUnqueriedColumns(self.base_table, columns,
                                                 operators, vals)
        cols, ops, vals, dominant_ops = ProjectQuery(self.fact_table, cols, ops,
                                                     vals)
        self.dominant_ops = dominant_ops
        return cols, ops, vals

    def _StandardizeQueryTree(self, query: dict, use_raw_table: bool):
        self.original_query = query
        cols, trees = FillInUnqueriedColumnsTree(self.base_table,
                                                 query)
        cols, ops, vals, dominant_ops = ProjectQuery(self.fact_table, cols, None,
                                                     None, trees, use_raw_table)
        self.dominant_ops = dominant_ops
        return cols, ops, vals, dominant_ops

    @profile
    def _get_valids(self, distinct_values, op, val, natural_idx, num_samples):
        has_mask = False
        # Column i.
        if op is not None:
            # There exists a filter.
            if self.fact_table.columns[natural_idx].factor_id is None:
                num_classes = len(distinct_values)
                # This column is not factorized.
                distinct_values, removed_nan = self._maybe_remove_nan(
                    distinct_values)
                if val is None:
                    assert False
                    valid = _tree_valids(distinct_values, op)
                else:
                    if not use_gpu_valid:
                        valids = [OPS[o](distinct_values, v) for o, v in zip(op, val)]
                        valid = np.logical_and.reduce(valids, 0).astype(np.float32,
                                                                        copy=False)
                    else:
                        if len(op) > 1:
                            #print('op', op)
                            #print('val', val)
                            #print('removed_nan', removed_nan)
                            #print('is ndarray', [isinstance(v, np.ndarray) for v in val])
                            #print('shape', [len(v.shape) if isinstance(v, np.ndarray) else None for v in val])
                            valids = [torch.tensor(OPS[o](distinct_values,
                                                          v[1:] if (removed_nan and isinstance(v, np.ndarray) and v.shape[0] == len(distinct_values) + 1) else v),
                                                   device=self.device).unsqueeze(0) for o, v in zip(op, val)]
                            #print('all_dvs', len(distinct_values))
                            #print('dvs[0]', distinct_values[0])
                            #print('valids', [valid.shape for valid in valids])
                            cat = torch.cat(valids)
                            if cat.shape[-1] > 0:
                                valid = cat.min(0).values
                            else:
                                valid = torch.tensor([], device=self.device)
                        else:
                            assert len(op) == 1
                            valid = torch.tensor(OPS[op[0]](distinct_values, val[0]), device=self.device)
                        if valid.shape[0] > len(distinct_values):
                            removed_nan = False
                if removed_nan:
                    if val is None:
                        assert False
                        v = 1. if op.op in ['IS_NULL'] else 0.
                    else:
                        v = 1. if op == ['IS_NULL'] else 0.
                    if use_gpu_valid:
                        valid = torch.cat([torch.tensor([v], device=self.device), valid])
                    else:
                        valid = np.insert(valid, 0, v)
                assert valid.shape == (num_classes,), f'valid {valid.shape} != {num_classes}\nwhere valid {valid} and\ndvs {distinct_values}'
            else:
                if debug or debug_get_valids:
                    print(f'(get_valids) natural_idx {natural_idx} factorized! # NDV =', len(distinct_values))
                if use_gpu_valid:
                    valid = None
                else:
                    valid = np.ones((len(op), len(distinct_values)), np.bool)
                for i, (o, v) in enumerate(zip(op, val)):
                    if use_gpu_valid:
                        valid_i = None
                    if o in common.PROJECT_OPERATORS.values(
                    ) or o in common.PROJECT_OPERATORS_LAST.values():
                        if use_gpu_valid:
                            valid_i = torch.tensor(OPS[o](distinct_values, v), device=self.device)
                        else:
                            valid[i] &= OPS[o](distinct_values, v)
                        has_mask = True
                        if self.fact_table.columns[natural_idx].factor_id > 0:
                            if use_gpu_valid:
                                assert valid_i is not None
                                if len(valid_i.shape) != 2:
                                    valid_i = valid_i.unsqueeze(0).expand(num_samples, -1)
                                assert valid_i.shape == (num_samples, len(distinct_values))
                                expanded_mask = self.factor_mask[i].unsqueeze(1)
                            else:
                                if len(valid.shape) != 3:
                                    valid = np.tile(np.expand_dims(valid, 1),
                                                    (1, num_samples, 1))
                                assert valid.shape == (len(op), num_samples,
                                                       len(distinct_values))
                                expanded_mask = np.expand_dims(
                                    self.factor_mask[i], 1)

                            assert expanded_mask.shape == (num_samples, 1)
                            if use_gpu_valid:
                                valid_i = torch.logical_or(valid_i, expanded_mask)
                            else:
                                valid[i] |= expanded_mask
                        if self.fact_table.columns[natural_idx].factor_id == 0:
                            if use_gpu_valid:
                                assert valid_i.shape == (len(distinct_values),)
                            else:
                                assert valid[i].shape == (len(distinct_values),), f'{valid[i].shape} not ({len(distinct_values)},)'
                        else:
                            if use_gpu_valid:
                                assert valid_i.shape == (num_samples,
                                                         len(distinct_values),)
                            else:
                                assert valid[i].shape == (num_samples,
                                                          len(distinct_values),), f'{valid[i].shape} not ({num_samples}, {len(distinct_values)})'
                    # IN is special case.
                    elif o == 'IN':
                        has_mask = True
                        if use_gpu_valid:
                            if no_repeat_large_v_list:
                                assert torch.is_tensor(v)
                                assert v.shape[-1] == len(distinct_values)
                            else:
                                v_list = torch.tensor(v, device=self.device)
                                matches = torch.tensor(distinct_values[:, None], device=self.device) == v_list
                        else:
                            v_list = np.array(list(v))
                            matches = distinct_values[:, None] == v_list
                        if not use_gpu_valid or not no_repeat_large_v_list:
                            assert matches.shape == (len(distinct_values),
                                                     len(v_list)), matches.shape
                        if self.fact_table.columns[natural_idx].factor_id > 0:
                            if use_gpu_valid:
                                if not no_repeat_large_v_list:
                                    matches = matches.expand(num_samples, -1, -1)
                                    expanded_mask = self.factor_mask[i].unsqueeze(1)
                            else:
                                if len(valid.shape) != 3:
                                    valid = np.tile(np.expand_dims(valid, 1),
                                                    (1, num_samples, 1))
                                assert valid.shape == (
                                    len(op), num_samples,
                                    len(distinct_values)), valid.shape
                                matches = np.tile(matches, (num_samples, 1, 1))
                                expanded_mask = np.expand_dims(
                                    self.factor_mask[i], 1)

                            if not use_gpu_valid or not no_repeat_large_v_list:
                                assert expanded_mask.shape == (num_samples, 1, len(v_list)), f'{expanded_mask.shape} not ({num_samples}, 1, {len(v_list)})'
                            if use_gpu_valid:
                                if not no_repeat_large_v_list:
                                    matches = torch.logical_and(matches, expanded_mask)
                            else:
                                matches &= expanded_mask

                            if not use_gpu_valid or not no_repeat_large_v_list:
                                assert matches.shape == (num_samples, len(distinct_values),
                                                         len(v_list)), f'{matches.shape} not ({num_samples}, {len(distinct_values)}, {len(v_list)})'
                        if use_gpu_valid:
                            assert valid_i is None
                            if no_repeat_large_v_list:
                                if self.fact_table.columns[natural_idx].factor_id == 0:
                                    valid_i = v
                                else:
                                    assert self.fact_table.columns[natural_idx].factor_id == 1
                                    assert len(v.shape) == 2
                                    assert torch.is_tensor(self.factor_mask[i])
                                    assert self.factor_mask[i].shape == (num_samples,)
                                    valid_i = v[self.factor_mask[i]]
                            else:
                                valid_i = torch.max(matches, -1).values #matches.max(-1)
                            if self.fact_table.columns[natural_idx].factor_id == 0:
                                assert valid_i.shape == (len(distinct_values),), f'{valid_i.shape} not ({len(distinct_values)},)'
                            else:
                                assert valid_i.shape == (num_samples,
                                                          len(distinct_values),), f'{valid_i.shape} not ({num_samples}, {len(distinct_values)})'
                        else:
                            valid[i] = np.logical_or.reduce(
                                matches, axis=-1).astype(np.float32, copy=False)
                            if self.fact_table.columns[natural_idx].factor_id == 0:
                                assert valid[i].shape == (len(distinct_values),), f'{valid[i].shape} not ({len(distinct_values)},)'
                            else:
                                assert valid[i].shape == (num_samples,
                                                          len(distinct_values),), f'{valid[i].shape} not ({num_samples}, {len(distinct_values)})'
                    else:
                        assert o != "NOT_IN" and o != "NOT IN"
                        if use_gpu_valid:
                            assert valid_i is None
                            valid_i = torch.tensor(OPS[o](distinct_values, v), device=self.device)
                        else:
                            valid[i] &= OPS[o](distinct_values, v)
                    if use_gpu_valid:
                        if i == 0:
                            assert valid is None
                            assert valid_i is not None
                            valid = valid_i
                        else:
                            assert valid is not None
                            assert valid_i is not None
                            valid = torch.logical_and(valid, valid_i)
                if not use_gpu_valid:
                    valid = np.logical_and.reduce(valid, 0).astype(np.float32,
                                                                   copy=False)
                if self.fact_table.columns[natural_idx].factor_id is None:
                    assert valid.shape == (len(distinct_values),), valid.shape
                else:
                    assert valid.shape == (num_samples, len(distinct_values)) or valid.shape == (len(distinct_values),), f'natural_idx {natural_idx} factor_id {self.fact_table.columns[natural_idx].factor_id} valid.shape {valid.shape}'
        else:
            # This column is unqueried.  All values are valid.
            valid = 1.0

        # Reset the factor mask if this col is not factorized
        # or if this col is the first subvar
        # or if we don't need to maintain a mask for this predicate.
        if self.fact_table.columns[natural_idx].factor_id in [None, 0
                                                             ] or not has_mask:
            self.factor_mask = None

        return valid, has_mask

    @profile
    def update_factor_mask(self, s, val, natural_idx):
        """Updates the factor mask for the next iteration.

        Factor mask is a list of length len(ops).  Each element in the list is
        a numpy array of shape (N, ?)  where the second dimension can be
        different sizes for different operators, indicating where a previous
        factor dominates the remaining for a column.

        We keep a separate mask for each operator for cases where there are
        multiple conditions on the same col. In these cases, there can be
        situations where a subvar dominates or un-dominates for one condition
        but not for others.

        The reason we need special handling is for cases like this: Let's say
        we have factored column x = (x1, x2) and literal y = (y1, y2) and the
        predicate is x > y.  By default we assume conjunction between subvars
        and so x>y would be treated as x1>y1 and x2>y2, which is incorrect.

        The correct handling would be
        x > y iff
            (x1 >= y1 and x2 > y2) OR
            x1 > y1.
        Because x1 contains the most significant bits, if x1>y1, then x2 can be
        anything.

        For the general case where x is factorized as (x1...xN) and y is
        factorized as (y1...yN),
        x > y iff
            (x1 >= y1 and x2 >= y2 and ... xN > yN) OR
            x1 > y1 OR x2 > y2 ... OR x(N-1) > y(N-1).
        To handle this, as we perform progressive sampling, we apply a
        "dominant operator" (> for this example) to the samples, and keep a
        running factor_mask that gets OR'd for calculating future valid
        vals. This function handles this.

        Other Examples:
            factored column x = (x1, x2), literals y = (y1, y2), z = (z1, z2)
            x < y iff
                (x1 <= y1 and x2 < y2) OR
                x1 < y1.
            x >= y iff
                (x1 >= y1 and x2 >= y2) OR
                x1 > y1.
            x <= y iff
                (x1 <= y1 and x2 <= y2) OR
                x1 < y1.
            x != y iff
                (any(x1) and x2 != y2) OR
                x1 != y1

        IN predicates are handled differently because instead of a single
        literal, there is a list of values. For example,
        x IN [y, z] if
                (x1 == y1 and x2 == y2) OR
                (x1 == z1 and x2 == z2)

        This function is called after _get_valids().  Note that _get_valids()
        would access self.factor_mask only after the first subvar for a given
        var (natural_idx).  By that time self.factor_mask has been assigned
        during the first invocation of this function.  Field self.factor_mask
        is reset to None by _get_valids() whenever we move to a new
        original-space column, or when the column has no operators that need
        special handling.
        """
        if not use_gpu_valid:
            s = s.cpu().numpy()
        if self.factor_mask is None:
            self.factor_mask = [None] * len(self.dominant_ops[natural_idx])
        for i, (p_op_dominant,
                v) in enumerate(zip(self.dominant_ops[natural_idx], val)):
            assert p_op_dominant != "NOT_IN"
            if p_op_dominant == 'IN':
                if use_gpu_valid:
                    if no_repeat_large_v_list:
                        self.factor_mask[i] = s
                        continue
                    else:
                        v_list = torch.tensor(v, device=self.device)
                else:
                    v_list = list(v)
                new_mask = s[:, None] == v_list
                if debug or debug_get_valids:
                    print('new_mask', new_mask.shape, new_mask)
                assert new_mask.shape == (len(s), len(v_list)), new_mask.shape
                if self.factor_mask[i] is not None:
                    if use_gpu_valid:
                        new_mask = torch.logical_and(new_mask, self.factor_mask[i])
                    else:
                        new_mask &= self.factor_mask[i]
                    if debug:
                        print('new_mask after and', new_mask)
            elif p_op_dominant in common.PROJECT_OPERATORS_DOMINANT.values():
                assert "LIKE" not in p_op_dominant
                if use_gpu_valid:
                    assert torch.is_tensor(s)
                    new_mask = TOPS[p_op_dominant](s, v)
                    assert torch.is_tensor(new_mask)
                else:
                    new_mask = OPS[p_op_dominant](s, v)
                if debug:
                    print('new_mask', new_mask.shape, new_mask)
                if self.factor_mask[i] is not None:
                    if use_gpu_valid:
                        new_mask = torch.logical_or(new_mask, self.factor_mask[i])
                    else:
                        new_mask |= self.factor_mask[i]
                    if debug:
                        print('new_mask after or', new_mask)
            else:
                assert p_op_dominant is None, 'This dominant operator ({}) is not supported.'.format(
                    p_op_dominant)
                if use_gpu_valid:
                    new_mask = torch.zeros_like(s, dtype=np.bool, device=self.device)
                else:
                    new_mask = np.zeros_like(s, dtype=np.bool)
                if debug:
                    print('new_mask', new_mask)
                if self.factor_mask[i] is not None:
                    if use_gpu_valid:
                        new_mask = torch.logical_or(new_mask, self.factor_mask[i])
                    else:
                        new_mask |= self.factor_mask[i]
                    if debug:
                        print('new_mask after or', new_mask)
            self.factor_mask[i] = new_mask
