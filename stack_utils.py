import pickle5 as pickle
import sys

all_sub_plan_queries = 'stack_queries/all_sub_plan_queries_str.pkl'

def qindex_to_qname(index):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for template in [2,3]:
        for qno in range(1,101):
            qno = str(qno)
            qname = f'q{template}-{qno.zfill(3)}'
            if qname in all_pairs:
                if index == i:
                    return qname
                i += 1
    return None

def qname_to_qindex(qname):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for template in [2,3]:
        for qno in range(1,101):
            qno = str(qno)
            name = f'q{template}-{qno.zfill(3)}'
            if name in all_pairs:
                if qname == name:
                    return i
                i += 1
    return None
