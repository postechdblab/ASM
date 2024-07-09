import pickle5 as pickle
import sys

all_sub_plan_queries = 'stats_queries/all_sub_plan_queries_str.pkl'

def qindex_to_qname(index):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for qno in range(146):
        qno = str(qno)
        qname = f'STATS_CEB_{qno.zfill(3)}'
        if qname in all_pairs:
            if index == i:
                return qname
            i += 1
    return None

def qname_to_qindex(qname):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for qno in range(146):
        qno = str(qno)
        name = f'STATS_CEB_{qno.zfill(3)}'
        if name in all_pairs:
            if qname == name:
                return i
            i += 1
    return None
