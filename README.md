# ASM: Harmonizing Autoregressive Model, Sampling, and Multi-dimensional Statistics Merging for Cardinality Estimation

This README provides guidelines on setting up the environment, training and evaluation of ASM, and on how to download, import, and run benchmarks.
The code is based on [NeuroCard](https://github.com/neurocard/neurocard) and [FactorJoin](https://github.com/wuziniu/FactorJoin).

## Environment Setup for Model Training and Evaluation

### Docker Setup for Model Training and Evaluation

Clone the ASM repository and make a directory for datasets. For convenience, we explain running our scripts in a Docker container named "asm_test". 
Run a Docker container with the continuumio/miniconda3 image with the conda installed. 
Replace <shared_memory_size> with a sufficient memory size to run all scripts. (Check your dev/shm memory using df /dev/shm -h command)

```
git clone <git_repository_of_ASM>
# <path_to_asm> is the path to the cloned repository
cd <path_to_asm>
# see the below instructions for downloading the datasets
mkdir datasets
docker run -itd --network=host --gpus all --shm-size=<shared_memory_size> -v <path_to_asm>:/home --name asm_test continuumio/miniconda3 /bin/bash
```

### Conda Environment Setup (This task should be run inside the asm_test Docker container)

Inside the asm_test Docker container, set up the conda environment as follows:

```
cd /home
apt-get update
apt-get install zip unzip gcc g++
conda env create --file ASM.yml
conda activate ASM
```

## Environment Setup for End-to-End Performance Measurement

### Docker Setup for Hacked PostgreSQL 

Please refer to https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark for setting up the PostgreSQL v13.1 for measuring end-to-end performance.
We've packaged all setups into a Docker image, including the PostgreSQL knob settings optimized for in-memory execution as mentioned in the paper.

```
docker pull sigmod2024id403/pg13_hacked_for_ce_benchmark
docker run --name ce-benchmark -p 5432:5432 -v <path_to_asm>:/home -d sigmod2024id403/pg13_hacked_for_ce_benchmark
```

## Download and Import the Datasets

Original dataset link for each dataset:


 - IMDB-JOB : http://homepages.cwi.nl/~boncz/job/imdb.tgz
 - Stack : https://rmarcus.info/stack.html
 - STATS-CEB : https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark

### Download IMDB-JOB

```
cd <path_to_asm>/datasets
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16Z35DYO-MfT_ipyNKSg6J21ZG40_LPgk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16Z35DYO-MfT_ipyNKSg6J21ZG40_LPgk" -O imdb_dataset.zip && rm -rf /tmp/cookies.txt
unzip imdb_dataset.zip
rm imdb_dataset.zip
```
If you have any problem with downloading & unzipping the imdb_dataset.zip, please refer to the following Google drive link: https://drive.google.com/file/d/16Z35DYO-MfT_ipyNKSg6J21ZG40_LPgk/view

### Import IMDB-JOB into PostgreSQL Docker
This task should be run inside the ce-benchmark Docker container.

```
psql -d postgres -U postgres
create database imdb;
\c imdb
\i /home/datasets/imdb/imdb_schema.sql
\i /home/datasets/imdb/imdb_load.sql
\i /home/datasets/imdb/imdb_index.sql
```

### Download Stack

```
cd <path_to_asm>/datasets
mkdir stack
cd stack
wget https://www.dropbox.com/s/55bxfhilcu19i33/so_pg13 -d stack_archive
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S02CL-TtibKu-3DpVuRfSGKLo9fJm8_0' -O stack_schema.sql
```

Due to the large size of Stack dataset, we provide the original dataset link.
It does not contain CSV files but a PostgreSQL dump file for tables and indexes.

### Import Stack into PostgreSQL 
This task should be run inside the ce-benchmark Docker container.

```
psql -d postgres -U postgres
create database stack;
\q
pg_restore -U postgres -d stack -v /home/datasets/stack/so_pg13
psql -d stack -U postgres
\i /home/datasets/stack/stack_index.sql
```

### Download STATS-CEB

```
cd <path_to_asm>/datasets
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=177TYJxneu6eiaEX6Iz1By3HG-xquIH4H' -O stats_datasets.zip
unzip stats_datasets.zip
rm stats_datasets.zip
```

### Import STATS-CEB into PostgreSQL Docker
This task should be run inside the ce-benchmark Docker container.

```
psql -d postgres -U postgres
create database stats;
\c stats
\i /home/datasets/stats_original/stats.sql
\i /home/datasets/stats_original/stats_load.sql
\i /home/datasets/stats_original/stats_index.sql
```

## Workloads

Workloads are located at:

 - IMDB-JOB : job_queries/all_queries.sql
 - Stack : stack-queries/all_queries.sql
 - STATS-CEB : stats_queries/all_queries.sql

## Generate Meta Model (This task should be run inside the asm_test Docker container)

Use these scripts to generate a meta model for each dataset, which contains the schema information and global ordering of join keys.
The meta models will be created in the "meta_models" directory (see inside the scripts).
In addition, a directory will be created for each table in the "datasets", where each directory contains "table0.csv" that corresponds to the reordered table following the global order.

### IMDB-JOB
```
bash generate_imdb_model.sh
```

### Stack
```
bash generate_stack_model.sh
```

### STATS-CEB
```
bash generate_stats_model.sh
```

## Train AR Models (This task should be run inside the asm_test Docker container)

Use these scripts to train the autoregressive (AR) models for each dataset. These models are trained over the reordered tables above.
The AR models will be created in the "AR_models" directory (see inside the scripts).
Furthermore, the AR models for the "*_type" tables of JOB and "site" table of Stack are dummies (not used in the estimation); following the implementation of FactorJoin (https://github.com/wuziniu/FactorJoin), we implement the per-table statistics estimation over the original table if the table has less than 1000 rows.



### IMDB-JOB
```
bash generate_imdb_ar.sh
```

### Stack

```
bash generate_stack_ar.sh
```

### STATS-CEB
```
bash generate_stats_ar.sh
```

## Estimate (This task should be run inside the asm_test Docker container)

Use these scripts to estimate the cardinalities of sub-queries of all queries for each dataset.
Each script requires the directories for the meta model and AR models (see inside the scripts).
The query-wise results will be stored in the **'<benchmark>_CE/result.<query_name>'** (e.g., job_CE/result.29b).

### STATS-CEB
```
bash evaluate_stats_ar.sh
```

### IMDB-JOB
```
bash evaluate_imdb_ar.sh
```

### Stack

```
bash evaluate_stack_ar.sh
```

## End-to-End Performance Measurement

First, ensure the Docker environment for the modified PostgreSQL (ce-benchmark) is correctly set up.

Second, merge all the query-wise estimates into a single file, starting from the smallest-ID query, e.g., 1a for JOB. See the *_utils.py files how the queries are ordered.

Third, use the following command to transfer the estimates to the ce-benchmark Docker container.

```
sudo docker cp /path/to/estimates/<host_est_file> ce-benchmark:/var/lib/pgsql/13.1/data/<docker_est_file>
```

Finally, use the following command to run PostgreSQL with the provided estimates. Replace <datset_name> either imdb, stack or stats and replace <benchmark> either job, stack or stats.

```
python send_query.py --dataset <dataset_name>
       --method_name <docker_est_file>
       --query_file ./<benchmark>_queries/all_queries.sql
       --save_folder <path_to_save_folder>
```

For example,
```
python send_query.py --dataset imdb
       --method_name imdb_est.txt
       --query_file ./job_queries/all_queries.sql
       --save_folder ./job_result_plans
```

This task should be run inside the asm_test Docker container.

## Utilizing Pre-trained Models

Pre-trained models are readily accessible for use. Make the evaluation script refer to the pre-trained models. They can be found at the following link:
 - [Google Drive Link for Pre-trained Models](https://drive.google.com/drive/folders/1rGBPrsCncxrN149etPd1JkKOii4E70Hl?usp=share_link)

