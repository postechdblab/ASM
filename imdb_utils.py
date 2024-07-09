import pickle5 as pickle
import sys

all_sub_plan_queries = 'job_queries/all_sub_plan_queries_str.pkl'

def qindex_to_qname(index):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for qno in range(1,34):
        for suffix in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
            qname = f'{qno}{suffix}'
            if qname in all_pairs:
                if index == i:
                    return qname
                i += 1
    return None

def qname_to_qindex(qname):
    all_pairs = pickle.load(open(all_sub_plan_queries, "rb"))

    i = 0
    for qno in range(1,34):
        for suffix in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
            name = f'{qno}{suffix}'
            if name in all_pairs:
                if qname == name:
                    return i
                i += 1
    return None

#print(qname_to_qindex('16b'))
