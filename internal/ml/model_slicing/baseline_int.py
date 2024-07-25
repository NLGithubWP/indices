import os
import torch
import argparse
from src.model.sparsemax_verticalMoe import SliceModel, SparseMax_VerticalSAMS

print("Read from src.model.sparsemax_verticalMoe import SliceModel, SparseMax_VerticalSAMS")
import time
import psycopg2
from src.model.factory import initialize_model
from typing import Any, List, Dict, Tuple
import json

USER = "postgres"
HOST = "localhost"
PORT = "28814"
DB_NAME = "pg_extension"
PASSWOD = "1234"

time_dict = {
    "load_model": 0,
    "data_query_time": 0,
    "py_conver_to_tensor": 0,
    "tensor_to_gpu": 0,
    "py_compute": 0

}


def read_json(file_name):
    print(f"Loading {file_name}...")
    is_exist = os.path.exists(file_name)
    if is_exist:
        with open(file_name, 'r', encoding='utf-8') as readfile:
            data = json.load(readfile)
        return data
    else:
        print(f"{file_name} is not exist")
        return {}


def fetch_and_preprocess(conn, batch_size, database, with_join):
    cur = conn.cursor()
    # Select rows greater than last_id
    if with_join == 1:
        print("Using Join operations")
        if database == "census":
            cur.execute(f"""SELECT
                     l.id,
                     l.label,
                     l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10,
                     l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20,
                     r.col21, r.col22, r.col23, r.col24, r.col25, r.col26, r.col27, r.col28, r.col29, r.col30,
                     r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41
                 FROM
                     {database}_int_train_left l
                 JOIN
                     {database}_int_train_right r ON l.id = r.id limit {batch_size};""")
        if database == "credit":
            cur.execute(f"""SELECT
                    l.id,
                    l.label,
                    l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12,
                    r.col13, r.col14, r.col15, r.col16, r.col17, r.col18, r.col19, r.col20, r.col21, r.col22, r.col23
                 FROM
                     {database}_int_train_left l
                 JOIN
                     {database}_int_train_right r ON l.id = r.id limit {batch_size};""")

        if database == "diabetes":
            cur.execute(f"""SELECT
                         l.id,
                        l.label,
                        l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24,
                        r.col25, r.col26, r.col27, r.col28, r.col29, r.col30, r.col31, r.col32, r.col33, r.col34, r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48
                 FROM
                     {database}_int_train_left l
                 JOIN
                     {database}_int_train_right r ON l.id = r.id where col3=10 and col4=17 limit {batch_size};""")

        if database == "hcdr":
            cur.execute(f"""SELECT
                        l.id,
                        l.label,
                        l.col1, l.col2, l.col3, l.col4, l.col5, l.col6, l.col7, l.col8, l.col9, l.col10, l.col11, l.col12, l.col13, l.col14, l.col15, l.col16, l.col17, l.col18, l.col19, l.col20, l.col21, l.col22, l.col23, l.col24, l.col25, l.col26, l.col27, l.col28, l.col29, l.col30, l.col31, l.col32, l.col33, l.col34,
                        r.col35, r.col36, r.col37, r.col38, r.col39, r.col40, r.col41, r.col42, r.col43, r.col44, r.col45, r.col46, r.col47, r.col48, r.col49, r.col50, r.col51, r.col52, r.col53, r.col54, r.col55, r.col56, r.col57, r.col58, r.col59, r.col60, r.col61, r.col62, r.col63, r.col64, r.col65, r.col66, r.col67, r.col68, r.col69
                 FROM
                     {database}_int_train_left l
                 JOIN
                     {database}_int_train_right r ON l.id = r.id where col33=383 and col38 =425 limit {batch_size};""")
    else:
        cur.execute(f"SELECT * FROM {database}_int_train LIMIT {batch_size}")
        print(f"SELECT * FROM {database}_int_train LIMIT {batch_size}")
    rows = cur.fetchall()
    return rows


def pre_processing(mini_batch_data: List[Tuple]):
    """
    mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
    """
    feat_id = torch.LongTensor(mini_batch_data)
    print("feat_id = ", feat_id[:, 2:].size())
    return {'id': feat_id[:, 2:]}


def fetch_data(database, batch_size, with_join):
    global time_dict
    print("Data fetching ....")
    begin_time = time.time()
    with psycopg2.connect(database=DB_NAME, user=USER, host=HOST, port=PORT) as conn:
        rows = fetch_and_preprocess(conn, batch_size, database, with_join)
    time_dict["data_query_time"] += time.time() - begin_time
    print(f"Data fetching done {rows[0]}, size = {len(rows)}, type = {type(rows)}, {type(rows[0])}")

    print("Data preprocessing ....")
    begin_time = time.time()
    batch = pre_processing(rows)
    time_dict["py_conver_to_tensor"] += time.time() - begin_time
    print("Data preprocessing done")
    return batch


def load_model(tensorboard_path: str, device: str = "cuda"):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)

    net = initialize_model(model_config)

    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path, map_location=device)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config


def if_cuda_avaiable(device):
    if "cuda" in device:
        return True
    else:
        return False


def reload_argparse(file_path: str):
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re

    return argparse.Namespace(**d)


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str,
                    help="directory to model file")
parser.add_argument('--flag', '-p', action='store_true',
                    help="wehther to print profile")
parser.add_argument('--print_net', '--b', action='store_true',
                    help="print the structure of network")

parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--dataset', type=str, default="frappe")
parser.add_argument('--target_batch', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--with_join', type=int, default=1)
parser.add_argument('--col_cardinalities_file', type=str, default="path to the stored file")

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    flag = args.flag
    device = torch.device(args.device)
    print(path)
    load_time = time.time()
    net, config = load_model(path, args.device)
    net: SparseMax_VerticalSAMS = net
    config.workload = 'random'
    time_dict["load_model"] = time.time() - load_time

    print(config.workload)

    overall_query_latency = time.time()
    if config.net == "sparsemax_vertical_sams":
        alpha = net.sparsemax.alpha
        print(alpha)

    print()

    col_cardinalities = read_json(args.col_cardinalities_file)

    # todo: we only test SPJ using this condition.
    where_cond = {}
    if args.with_join == 1:
        if args.dataset == "diabetes":
            where_cond = {"2": 10, "3": 17}
        if args.dataset == "hcdr":
            where_cond = {"32": 383, "37": 425}

    target_sql_list = col_cardinalities
    for col_index, value in where_cond.items():
        target_sql_list[int(col_index)] = value
    target_sql = torch.tensor(target_sql_list).reshape(1, -1)

    net.eval()
    net = net.to(device)
    with torch.no_grad():
        sql = target_sql.to(device)
        if config.net == "sparsemax_vertical_sams":
            subnet: SliceModel = net.tailor_by_sql(sql)
            subnet.to(device)
        else:
            subnet = net
        subnet.eval()
        target_list, y_list = [], []
        ops = 0

        # default batch to 1024
        num_ite = args.target_batch // args.batch_size
        print(f"num_ite = {num_ite}")
        for i in range(num_ite):
            # fetch from db
            data_batch = fetch_data(args.dataset, args.batch_size, args.with_join)
            print("Copy to device")
            # wait for moving data to GPU
            begin = time.time()
            x_id = data_batch['id'].to(device)
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["tensor_to_gpu"] += time.time() - begin

            print(f"begin to compute on {args.device}, is_cuda = {if_cuda_avaiable(args.device)}")
            # compute
            begin = time.time()
            y = subnet(x_id, None)
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["py_compute"] += time.time() - begin
    time_dict["overall_query_latency"] = time.time() - overall_query_latency
    print(time_dict)
