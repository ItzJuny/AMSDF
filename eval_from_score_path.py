import argparse
import os
import numpy as np
import torch
from torch import nn
import pandas as pd
import warnings
from tools.tool_eval import *
warnings.filterwarnings("ignore")
__author__ = "JunyanWu"
__email__ = "wujy298@mail2.sysu.edu.cn"


def score_path_df(epath):
    filepaths=[os.path.join(epath,i) for i in os.listdir(epath)]
    df=pd.DataFrame()
    for fp in filepaths:
        if fp.endswith('.txt') ==False:
            continue
        ename=os.path.basename(fp).replace('.txt','')
        result=switch_case(ename, fp)
        temp_keys=['modelname', 'dataset']
        temp_keys.extend(result.keys())
        temp_values=[os.path.basename(epath),ename]
        temp_values.extend([float(value) for value in result.values()])
        temp_df=pd.DataFrame([temp_values],columns=temp_keys)
        df=pd.concat([df, temp_df], ignore_index=True)
    avg_values=[os.path.basename(epath),'eer_avg']
    avg_values.append(df['eer'].mean())
    avg_df=pd.DataFrame([avg_values],columns=['modelname', 'dataset','eer'])
    df=pd.concat([df, avg_df], ignore_index=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AMSDF')
    parser.add_argument('--score_path', type=str, default='./scores_AMSDF')
    args = parser.parse_args()
    # check socre path
    # score_file: ./scores/M1/asvs2019la.txt
    if args.score_path.endswith('.txt'):
        ename=os.path.basename(args.score_path).replace('.txt','')
        result=switch_case(ename, args.score_path)
        print(ename, evaleer_fromtxt(args.score_path),  result)
        sys.exit(0)
    filepaths=[os.path.join(args.score_path, i) for i in os.listdir(args.score_path)]
    df_all=pd.DataFrame()
    for fp in filepaths:
        # score_subpath: ./scores/M1/
        if fp.endswith('.txt'):
            df=score_path_df(args.score_path)
            print(args.score_path,'\n',df)
            df.to_csv(os.path.join(args.score_path, 'model_result.csv'))
            sys.exit(0)
        # score_rootpath: ./scores/
        elif os.path.isdir(fp):
            df=score_path_df(fp)
            print(fp,'\n',df)
            df.to_csv(os.path.join(fp, 'model_result.csv'))
            df_all=pd.concat([df_all, df], ignore_index=True)
    # all results statistic
    grouped=df_all.groupby(['dataset'])
    df_total=pd.DataFrame()
    for name, group in grouped:
        if name[0]=='eer_avg':
            continue
        total_values=['all_models',name[0]]
        total_values.append(group['eer'].mean())
        total_values.append(group['eer'].min())
        df=pd.DataFrame([total_values],columns=['modelname', 'dataset','avg_eer','best_eer'])
        df_total=pd.concat([df_total, df], ignore_index=True)
    print('all results statistic \n',df_total)
    df_total.sort_values(by='dataset')
    df_total.to_csv(os.path.join(args.score_path, 'all_model_results.csv'))