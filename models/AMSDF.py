import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from .ACRNN import acrnn
from .AASIST import *

class ASR_model(nn.Module):
    def __init__(self):
        super(ASR_model, self).__init__()
        cp_path = os.path.join('./pretrained_models/xlsr2_300m.pt')   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].cuda()
        self.linear = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        emb = self.linear(emb) 
        emb = F.max_pool2d(emb, (4,2)) 
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

    
class SER_model(nn.Module):
    def __init__(self):
        super(SER_model, self).__init__()
        cp_path = os.path.join('./pretrained_models/ser_acrnn.pth')   # Change the pre-trained SER model path. 
        model=acrnn().cuda()
        model.load_state_dict(torch.load(cp_path))
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)
        self.model = model

    def forward(self, x):
        emb = self.model(x)
        emb = F.max_pool2d(emb, (3, 4)) 
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

    
class base_encoder(nn.Module):
    def __init__(self):
        super(base_encoder, self).__init__()
        filts= [[1, 32], [32, 32], [32, 64], [64, 64]]
        self.conv_time=CONV(out_channels=70,
                              kernel_size=128,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[0], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[1])),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3]))) 

    def forward(self, x, Freq_aug):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        emb = self.encoder(x) 
        out, _ = torch.max(torch.abs(emb), dim=2) 
        out = out.transpose(1, 2) 
        return out


class HGFM(nn.Module):
    def __init__(self):
        super(HGFM, self).__init__()
        self.HtrgGAT_layer1 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.HtrgGAT_layer2 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.stack_node = nn.Parameter(torch.randn(1, 1, 64))

    def forward(self, x1, x2):
        stack_node = self.stack_node.expand(x1.size(0), -1, -1)
        x1, x2, stack_node,_ = self.HtrgGAT_layer1(x1, x2, master=stack_node)
        x1_aug, x2_aug, stack_node2,attmap = self.HtrgGAT_layer2(x1, x2, master=stack_node)
        x1 = x1 + x1_aug 
        x2 = x2 + x2_aug
        stack_node = stack_node + stack_node2 
        x1 = self.drop_way(x1)
        x2 = self.drop_way(x2)
        stack_node = self.drop_way(stack_node)
        return x1+x2, stack_node, attmap


class GRS(nn.Module):
    def __init__(self):
        super(GRS, self).__init__()
        self.pool1 = GraphPool(0.5, 64, 0.3)
        self.pool2 = GraphPool(0.5, 64, 0.3)
    def forward(self, x_list):
        pool_list=[]
        for i in x_list:
            pool_list.append(self.pool2(self.pool1(i)))
        pool_cat=torch.cat(pool_list, dim=1)
        pool_max, _=torch.max(torch.abs(pool_cat),dim=1)
        pool_avg=torch.mean(pool_cat,dim=1)
        return torch.cat([pool_max,pool_avg], dim=1)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        """multi-view feature extractor"""
        self.text_view_extract=ASR_model()
        self.emo_view_extract=SER_model()
        self.audio_view_extract=base_encoder()
        """IGAM"""
        self.GAT_text = GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_text = GraphPool(0.5, 64, 0.3)
        self.GAT_emo = GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_emo = GraphPool(0.5, 64, 0.3)
        self.GAT_audio= GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_audio = GraphPool(0.88, 64, 0.3)
        """HGFM"""
        self.Core_AE = HGFM()
        self.Core_AT = HGFM()
        self.Core_ET = HGFM()
        self.Core_AET = HGFM()
        """GRS"""
        self.GRS_group1=GRS()
        self.GRS_group2=GRS()
        self.GRS_group3=GRS()
        self.drop = nn.Dropout(0.5, inplace=True)
        self.out_layer = nn.Linear(384, 64)
        self.out_layer2 = nn.Linear(64, 2)
        
    def forward(self, inputs,inputs2, Freq_aug):
        x=inputs
        x2=inputs2
        """multi-view features"""
        audio_view=self.audio_view_extract(x, Freq_aug=Freq_aug) 
        emo_view=self.emo_view_extract(x2)
        text_view=self.text_view_extract(x)
        """ Intra-view graph attention module"""
        emo_gat = self.GAT_emo(emo_view) 
        audio_gat = self.GAT_audio(audio_view) 
        text_gat = self.GAT_text(text_view) 
        emo_gat = self.pool_emo(emo_gat) 
        audio_gat = self.pool_audio(audio_gat) 
        text_gat = self.pool_text(text_gat)
        """ Heterogeneous graph fusion module"""
        AE_HG, AE_SN,attmap_AE = self.Core_AE(audio_gat, emo_gat) # A-E
        AT_HG, AT_SN,attmap_AT = self.Core_AT(audio_gat, text_gat) # A-T
        ET_HG, ET_SN,attmap_ET = self.Core_ET(emo_gat, text_gat) # E-T
        AET_HG, AET_SN,attmap_AET = self.Core_AET(AE_HG, ET_HG) # A-E-T
        """Group-based Readout Scheme"""
        GAT_Group=[audio_gat,emo_gat,text_gat]
        HGAT_Group=[AE_HG,AT_HG,ET_HG,AET_HG]
        SN_Group=[AE_SN,AT_SN,ET_SN,AET_SN]
        out1=self.GRS_group1(GAT_Group)
        out2=self.GRS_group2(HGAT_Group)
        out3=self.GRS_group3(SN_Group)
        """output"""
        last_hidden = torch.cat([out1,out2,out3], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        output = self.out_layer2(output)
        return output
