from pprint import pprint
import argparse
import pandas as pd

from modules.metrics import compute_mlc
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/iu_xray/res_labeled.csv', help='the path to the directory containing the res data.')
    parser.add_argument('--gts_path', type=str, default='results/iu_xray/gts_labeled.csv', help='the path to the directory containing the gts data.')
    return parser.parse_args()

def main():
    args = parse_args()
    res_path, gts_path = args.res_path, args.gts_path
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    metrics = compute_mlc(gts_data, res_data, label_set)
    res_dir = res_path[:-len(res_path.split('/')[-1])]
    # print(res_dir)
    json.dump(metrics, open(f'{res_dir}/AURROC.json', 'w'))
    pprint(metrics)



if __name__ == '__main__':
    main()
