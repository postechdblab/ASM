import psycopg2
import time
import os
import argparse
import numpy as np

def send_query(dataset, method_name, query_file, save_folder, iteration=None):
    conn = psycopg2.connect(database=dataset, user="postgres", password="postgres", host="localhost", port=5432)
    conn.set_client_encoding('UTF8')
    cursor = conn.cursor()

    with open(query_file, "r") as f:
        queries = f.readlines()

    # cursor.execute('SET debug_card_est=true')
    # cursor.execute('SET print_sub_queries=true')
    # cursor.execute('SET print_single_tbl_queries=true')
    cursor.execute("SET statement_timeout = 0; commit;")

    ##### ML METHOD OPTIONS #####

    cursor.execute("SET ml_joinest_enabled=true;")
    cursor.execute("SET join_est_no=0;")
    cursor.execute(f"SET ml_joinest_fname='{method_name}';")
        

    cursor.execute("SET enable_indexscan=on;")
    cursor.execute("SET enable_bitmapscan=on;")
    
    cursor.execute("SET enable_material=off;")

    cursor.execute("SET enable_hashjoin = on;")
    cursor.execute("SHOW enable_hashjoin;")
    print("enable_hashjoin : ", cursor.fetchall())

    cursor.execute("SHOW ml_cardest_enabled;")
    print("ml_cardest_enabled : ", cursor.fetchall())
    
    cursor.execute("SHOW ml_joinest_enabled;")
    print("ml_joinest_enabled : ", cursor.fetchall())
    cursor.execute("SHOW ml_joinest_fname;")
    print("ml_joinest_fname :", cursor.fetchall())
    cursor.execute("SHOW join_est_no;")
    print("join_est_no :", cursor.fetchall())
    cursor.execute("SHOW debug_card_est;")
    print("debug_card_est :", cursor.fetchall())
    cursor.execute("SHOW print_sub_queries;")
    print("print_sub_queries : ", cursor.fetchall())
    cursor.execute("SHOW print_single_tbl_queries;")
    print("print_single_tbl_queries : ", cursor.fetchall())

    cursor.execute("SHOW enable_indexonlyscan;")
    print("enable_indexonlyscan : ", cursor.fetchall())

    cursor.execute("SHOW print_single_tbl_queries;")
    print("print_single_tbl_queries : ", cursor.fetchall())

    cursor.execute("SHOW enable_material;")
    print("enable_material : ", cursor.fetchall())

    cursor.execute("SHOW enable_indexscan;")
    print("enable_indexscan : ", cursor.fetchall())

    cursor.execute("SHOW enable_indexonlyscan;")
    print("enable_indexonlyscan : ", cursor.fetchall())

    cursor.execute("SHOW enable_nestloop")
    print("enable_nestloop : ", cursor.fetchall())

    cursor.execute("SHOW enable_hashagg")
    print("enable_hashagg : ", cursor.fetchall())

    cursor.execute("SHOW enable_bitmapscan")
    print("enable_bitmapscan : ", cursor.fetchall())

    cursor.execute("SHOW ml_cardest_enabled;")
    print("ml_cardest_enabled : ", cursor.fetchall())

    cursor.execute("SHOW ml_cardest_fname;")
    print("ml_cardest_fname : ", cursor.fetchall())

    cursor.execute("SHOW work_mem;")
    print("work_mem : ", cursor.fetchall())

    cursor.execute("SHOW shared_buffers;")
    print("shared_buffers : ", cursor.fetchall())

    cursor.execute("SHOW effective_cache_size;")
    print("effective_cache_size : ", cursor.fetchall())

    cursor.execute("SHOW geqo_threshold;")
    print("geqo_threshold : ", cursor.fetchall())

    planning_time = [] 
    execution_time = []

    cumulative_time = 0

    save_file_name = method_name.split(".txt")[0]

    num_total_queries = len(queries)
    timeout_queries_no = []

    for no, query in enumerate(queries):
        
        if "||" in query and dataset != "stats":
            query = query.split("||")[-1]
        elif "||" in query and dataset == "stats":
            query = query.split("||")[1]
        
        query = query.strip()

        print(f"Executing query {no}")
        start = time.time()

        res = None

        try:
            cursor.execute("EXPLAIN ANALYZE " + query)
            res = cursor.fetchall()
            planning_time.append(float(res[-2][0].split(":")[-1].split("ms")[0].strip()))
            execution_time.append(float(res[-1][0].split(":")[-1].split("ms")[0].strip()))
        except Exception as err:
            print(err)
            print("Type of error : ", type(err))
            timeout_queries_no.append(no)
            planning_time.append(0.0)
            execution_time.append(0.0)

        filename = "query_" + str(no) + "_plan.txt"
        if iteration is not None: 
            directory = save_folder + '/result_plan_iter_' + str(iteration) + '/'
        else :  
            directory = save_folder + f'/result_plan_{save_file_name}/'

        os.makedirs(directory, exist_ok=True)
        
        with open(directory+filename, 'w') as file:
            if res is not None:
                for item in res:
                    file.write(item[0] + '\n')
            else :
                file.write("None\n")
        
        end = time.time()
        print(f"{no}-th query finished in {end-start} sec, with planning_time {planning_time[no]} ms and execution_time {execution_time[no]} ms" )
        # print expected time
        cumulative_time += (end - start)
        print("Expected Remaining Time by Extrapolation (sec) :", (num_total_queries - no - 1) * (cumulative_time / (no + 1)))
        print("Timeout Queries Number:", timeout_queries_no)

    cursor.close()
    conn.close()
    
    if iteration:
        np.save(save_folder + f"plan_time_{save_file_name}_iter{iteration}", np.asarray(planning_time))
        np.save(save_folder + f"exec_time_{save_file_name}_iter{iteration}", np.asarray(execution_time))
    else:
        np.save(save_folder + f"plan_time_{save_file_name}", np.asarray(planning_time))
        np.save(save_folder + f"exec_time_{save_file_name}", np.asarray(execution_time))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stats', help='Which dataset to be used')
    parser.add_argument('--method_name', default='stats_CEB_sub_queries_model_stats_greedy_50.txt', help='save estimates')
    parser.add_argument('--query_file', default='/home/ubuntu/data_CE/stats_CEB/stats_CEB.sql', help='Query file location')
    parser.add_argument('--save_folder', default='/home/ubuntu/data_CE/stats_CEB/', help='Query file location')
    parser.add_argument('--iteration', type=int, default=None, help='Number of iteration to read')

    args = parser.parse_args()

    if args.iteration:
        for i in range(args.iteration):
            send_query(args.dataset, args.method_name, args.query_file, args.save_folder, i+1)
    else:
        send_query(args.dataset, args.method_name, args.query_file, args.save_folder, None)