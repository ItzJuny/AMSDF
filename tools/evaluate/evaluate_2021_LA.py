#!/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has tje CM protocol and ASV score.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py score.txt ./keys eval
"""

import sys, os.path
import numpy as np
import pandas 
from glob import glob
from . import eval_metric_LA as em






def load_asv_metrics(phase,asv_key_file,asv_scr_file):
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)########

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
        'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
        'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
    }
    bona_cm = cm_scores[cm_scores[5]=='bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5]=='spoof']['1_x'].values
    bona_cm=np.array(bona_cm,dtype=float)
    spoof_cm=np.array(spoof_cm,dtype=float)
    # If label reverse
    invert=False
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    eer_cm_re = em.compute_eer(-bona_cm, -spoof_cm)[0]
    if eer_cm_re<eer_cm:
        invert=True
        eer_cm=eer_cm_re
    if invert==False:
        tDCF_curve, _ = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = em.compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_to_score_file(phase, score_file, asv_key_file,asv_scr_file, cm_key_file):
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(phase, asv_key_file, asv_scr_file)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    scores_np = np.genfromtxt(score_file, dtype=str)[:,[0,1]]
    submission_scores = pandas.DataFrame(scores_np)
    uid=np.array(submission_scores.loc[:,0])
    uid=[os.path.splitext(os.path.basename(i))[0] for i in uid]
    submission_scores.loc[:,0]=uid
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    if len(cm_scores) != np.sum(cm_data[7] == phase):
        print('CHECK: submission has %d of %d expected trials.' % (len(cm_scores), np.sum(cm_data[7] == phase)))
        exit(1)
    # print(cm_scores)
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100*eer_cm)

    return eer_cm*100,min_tDCF


