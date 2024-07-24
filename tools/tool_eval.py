"""eval function"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve,confusion_matrix,auc,roc_curve
from .evaluate.evaluate_2019 import calculate_tDCF_EER as eval_19LA
from .evaluate.evaluate_2021_LA import eval_to_score_file as eval_21LA
from .evaluate.evaluate_2021_DF import eval_to_score_file as eval_21DF
__author__ = "JunyanWu"
__email__ = "wujy298@mail2.sysu.edu.cn"


def eval2019LA_fromtxt(scorefile):
    protocol_path=os.path.join('tools/evaluate/keys/', "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
    asv_path=os.path.join('tools/evaluate/keys/', "ASVspoof2019_LA_asv_scores", 'ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')
    return eval_19LA(scorefile,protocol_path,asv_path)


def evaleer_fromtxt(scorefile):
    content=np.loadtxt(scorefile,dtype=str)
    score_np=content[:,1]
    label_np=content[:,2]
    eer=compute_eer(label_np,score_np)
    return {"eer":"%.4f"%eer}


def _compute_eer(label_np, score_np):
    frr, far, thresholds = det_curve(label_np, score_np)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer=np.mean((frr[min_index], far[min_index]))
    return eer

def compute_eer(label_np, score_np):
    score_np=np.array(score_np,dtype=float)
    label_np=np.array(label_np,dtype=float)
    # check if label reverse
    label_np_re=1-label_np
    eer,err_re=_compute_eer(label_np, score_np),_compute_eer(label_np_re, score_np)
    return min(eer,err_re)*100

def eval2021LA_fromtxt(scorefile):
    asv_key_file = os.path.join("tools/evaluate/keys/LA-keys-stage-1/keys/", 'ASV/trial_metadata.txt')
    asv_scr_file = os.path.join("tools/evaluate/keys/LA-keys-stage-1/keys/", 'ASV/ASVTorch_Kaldi/score.txt')
    cm_key_file = os.path.join("tools/evaluate/keys/LA-keys-stage-1/keys/", 'CM/trial_metadata.txt')
    eer,mtdcf = eval_21LA('eval',scorefile, asv_key_file,asv_scr_file, cm_key_file)
    return {"eer":"%.4f"%eer,"mtdcf":"%.4f"%mtdcf}


def eval2021DF_fromtxt(scorefile):
    cm_key_file = os.path.join("tools/evaluate/keys/DF-keys-stage-1/keys/", 'CM/trial_metadata.txt')
    eer = eval_21DF(phase='eval',score_file=scorefile, cm_key_file=cm_key_file)
    return {"eer":"%.4f"%eer}



def switch_case(name, *args):
    switch_dict = {
        'asvs2019la': (eval2019LA_fromtxt,*args),
        'asvs2021la': (eval2021LA_fromtxt,*args),
        'asvs2021df': (eval2021DF_fromtxt,*args),
        'others': (evaleer_fromtxt,*args),
    }
    selected_case, *case_args = switch_dict.get(name, switch_dict['others'])
    return selected_case(*case_args)