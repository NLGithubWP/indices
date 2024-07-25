

import argparse
import os
import glob
import json
from src.data_loader import SQLAttacedLibsvmDataset




parser = argparse.ArgumentParser(description="Process some dataset parameters.")
parser.add_argument('--data_dir', type=str, required=True, help='Directory where the data is stored')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--nfield', type=int, required=True, help='Number of fields')
parser.add_argument('--max_filter_col', type=int, default= 4, help='Maximum number of filter columns')
parser.add_argument('--out_file', type=str, default="./")
args = parser.parse_args()



'''Command
python save_satistics.py --data_dir /hdd1/sams/data/ \
    --dataset avazu --nfield 22 \
    --out_file ./avazu_padding.json
    
python save_satistics.py --data_dir /hdd1/sams/data/ \
    --dataset census --nfield 41 \
    --out_file ./census_padding.json

python save_satistics.py --data_dir /hdd1/sams/data/ \
    --dataset hcdr --nfield 69 \
    --out_file ./hcdr_padding.json
    
python save_satistics.py --data_dir /hdd1/sams/data/ \
    --dataset diabetes --nfield 48 \
    --out_file ./diabetes_padding.json

python save_satistics.py --data_dir /hdd1/sams/data/ \
    --dataset credit --nfield 23 \
    --out_file ./credit_padding.json
'''
if __name__ == '__main__':
    
    data_dir = os.path.join(args.data_dir, args.dataset)
    train_file  = glob.glob("%s/train.*" % data_dir)[0]
    dataset = SQLAttacedLibsvmDataset(train_file, args.nfield, args.max_filter_col)
    
    # List: col_idx -> padding_token
    padding_list = dataset.padding_feature_id
    
    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump(padding_list, f)
