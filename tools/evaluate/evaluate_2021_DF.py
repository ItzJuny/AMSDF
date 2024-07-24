#!/usr/bin/env python
"""
Script to compute pooled EER for ASVspoof2021 DF. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py score.txt ./keys eval
"""
import os
import numpy as np
import pandas
from . import eval_metrics_DF as em
from glob import glob



def eval_to_score_file(phase, score_file,  cm_key_file):
   cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
   scores_np = np.genfromtxt(score_file, dtype=str)[:,[0,1]]
   submission_scores = pandas.DataFrame(scores_np)
   uid=np.array(submission_scores.loc[:,0])
   uid=[os.path.splitext(os.path.basename(i))[0] for i in uid]
   submission_scores.loc[:,0]=uid
   cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
   bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
   spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
   bona_cm = np.array(bona_cm,dtype=float)
   spoof_cm =np.array(spoof_cm,dtype=float)
   eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
   # Check label reverse
   eer_cm_re= em.compute_eer(-bona_cm, -spoof_cm)[0]
   eer_cm=min(eer_cm, eer_cm_re)
   return eer_cm*100

