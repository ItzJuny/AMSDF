import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from models.AMSDF import Module
from data_io import get_dataloader
from tools.startup_config import set_random_seed
import warnings
warnings.filterwarnings("ignore")
from main_AMSDF import *
__author__ = "JunyanWu"
__email__ = "wujy298@mail2.sysu.edu.cn"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python inference_all_dataset.py -bs 64 --emodel xx.pth')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--emodel', type=str, default="./checkpoints/best/e6_lr0.000001_eer2.00.pth")
    parser.add_argument('--score_path', type=str, default="./scores")
    parser.add_argument('--fresh', action='store_true', default=False)
    args = parser.parse_args()
    set_random_seed(args.seed)
    model = Module().cuda()
    print('Model loaded : {}'.format(args.emodel))
    model.load_state_dict(torch.load(args.emodel), False)
    for ename in ['asvs2019la', 'asvs2021la','asvs2021df','for','fac','itw','asvs2015e','asvs2015d']:
        score_file=os.path.join(args.score_path,'%s'%os.path.basename(args.emodel).replace(".pth",""),"%s.txt"%(ename))
        os.makedirs(os.path.dirname(score_file),exist_ok=True)
        if os.path.exists(score_file) and not args.fresh:
            continue
        eval_dlr = get_dataloader(batch_size=args.batch_size,num_workers=args.num_workers, if_eval=True, ename=ename)
        eval_eer=produce_evaluation_file(eval_dlr, model, save_path=score_file)
        print(ename,eval_eer, '\n saving to', score_file)
        del eval_dlr
