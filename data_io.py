import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import python_speech_features as ps
from tqdm import tqdm
import librosa
REAL=1
FAKE=0
__author__ = "JunyanWu"
__email__ = "wujy298@mail2.sysu.edu.cn"


class ASVspoof2019LA(Dataset):
    def __init__(self,part,  path="/data/wujy/audio/asvspoof/ASVspoof2019_LA"):
        self.part = part
        if self.part == "train":
            protocol_path=os.path.join(path, "ASVspoof2019_LA_cm_protocols","ASVspoof2019.LA.cm.train.trn.txt")
        elif self.part == "dev":
            protocol_path=os.path.join(path, "ASVspoof2019_LA_cm_protocols","ASVspoof2019.LA.cm.dev.trl.txt")
        elif self.part == "eval":
            protocol_path=os.path.join(path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
        self.label_dict={"bonafide":REAL,"spoof":FAKE}
        self.protocol = np.loadtxt(protocol_path,dtype=str)
        self.path = [path for i in range(len(self.protocol))]
        # attack types
        self.spoof2fn_dict={}
        for i in range(len(self.protocol)):
            spoof_label=self.protocol[i,-2] if self.protocol[i,-2]!='-' else 'bonafide'
            filename=self.protocol[i,1]
            if spoof_label not in self.spoof2fn_dict:
                self.spoof2fn_dict[spoof_label]=[filename]
            else:
                self.spoof2fn_dict[spoof_label].append(filename)
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        filename = protocol_idx[1]
        filepath = os.path.join(self.path[idx],"ASVspoof2019_LA_%s"%self.part, 'flac',filename + ".flac")
        label = torch.tensor(self.label_dict[protocol_idx[4]], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename
    
    def __len__(self):
        return len(self.protocol)
    
    def load_audio(self,filepath):
        wave, fs = librosa.load(filepath, sr=16000)
        return np.array(wave,dtype=float)

    
    def extract_mel(self,x):
        begin, end=0, 300
        mel_spec = ps.logfbank(x, 16000, nfilt = 40)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        fea=[mel_spec[begin:end,:],delta1[begin:end,:],delta2[begin:end,:]]
        return np.array(fea,dtype=float)
    
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return np.array(padded_x,dtype=float)

    
    def spoof_label_dict(self):
        ans={}
        count=0
        spooflabels=sorted(self.spoof2fn_dict.keys())
        for sl in spooflabels:
            ans[sl]=self.spoof2fn_dict[sl]
            count+=len(self.spoof2fn_dict[sl])
        return ans
 
    
class ASVspoof2021LA(ASVspoof2019LA):
    def __init__(self, part,path="/data/wujy/audio/asvspoof"):
        self.label_dict={"bonafide":REAL, "spoof":FAKE}
        self.path=os.path.join(path,"ASVspoof2021_LA_eval")
        protocol_path="./tools/evaluate/keys/LA-keys-stage-1/keys/CM/trial_metadata.txt"
        #ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt
        protocol = np.loadtxt(protocol_path,dtype=str)
        if part=="eval":
            protocol_mask=(protocol[:,7]=="eval")
            self.protocol=protocol[protocol_mask]
        attack2spooflabel={
            "none":"L01",
            "alaw":"L02",
            "pstn":"L03",
            "g722":"L04",
            "ulaw":"L05",
            "gsm":"L06",
            "opus":"L07",
        }
        attacks=self.protocol[:,2]
        self.spoof2fn_dict={}
        for i in range(len(self.protocol)):
            spoof_label=attack2spooflabel[attacks[i]]
            filename=self.protocol[i,1]
            if spoof_label not in self.spoof2fn_dict:
                self.spoof2fn_dict[spoof_label]=[filename]
            else:
                self.spoof2fn_dict[spoof_label].append(filename)
    
    
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        filename=protocol_idx[1]
        filepath = os.path.join(self.path,'flac',filename + ".flac")
        label = torch.tensor(self.label_dict[protocol_idx[5]], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename
    def spoof_label_dict(self):
        ans={}
        count=0
        spooflabels=sorted(self.spoof2fn_dict.keys())
        for sl in spooflabels:
            ans[sl]=self.spoof2fn_dict[sl]
            count+=len(self.spoof2fn_dict[sl])
        return ans

class ASVspoof2021DF(ASVspoof2019LA):
    def __init__(self, part,path="/data/wujy/audio/asvspoof"):
        self.label_dict={"bonafide":REAL, "spoof":FAKE}
        self.path=os.path.join(path,"ASVspoof2021_DF_eval")
        protocol_path="./tools/evaluate/keys/DF-keys-stage-1/keys/CM/trial_metadata.txt"
        #"ASVspoof2021_DF_eval/keys/DF/CM/trial_metadata.txt"
        protocol = np.loadtxt(protocol_path,dtype=str)
        if part=="eval":
            protocol_mask=(protocol[:,7]=="eval")
            self.protocol=protocol[protocol_mask]
        attack2spooflabel={
            "bonafide":"bonafide",
            "nocodec":"D01",
            "low_mp3":"D02",
            "high_mp3":"D03",
            "low_m4a":"D04",
            "high_m4a":"D05",
            "low_ogg":"D06",
            "high_ogg":"D07",
            "mp3m4a":"D08",
            "oggm4a":"D09",
        }
        attacks=self.protocol[:,2]
        self.spoof2fn_dict={}
        for i in range(len(self.protocol)):
            spoof_label=attack2spooflabel[attacks[i]]
            filename=self.protocol[i,1]
            if spoof_label not in self.spoof2fn_dict:
                self.spoof2fn_dict[spoof_label]=[filename]
            else:
                self.spoof2fn_dict[spoof_label].append(filename)
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        filename=protocol_idx[1]
        filepath = os.path.join(self.path,'flac',filename + ".flac")
        label = torch.tensor(self.label_dict[protocol_idx[5]], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename
    def spoof_label_dict(self):
        count=0
        ans={}
        spooflabels=sorted(self.spoof2fn_dict.keys())
        for sl in spooflabels:
            ans[sl]=self.spoof2fn_dict[sl]
            count+=len(self.spoof2fn_dict[sl])
        return ans
    
class ASVspoof2015(ASVspoof2019LA):
    def __init__(self,part,path="/data/wujy/audio/asvspoof/ASVspoof2015" ):
        self.part = part
        if self.part == "train":
            protocol_path=os.path.join(path, "CM_protocol","cm_train.trn")
        elif self.part == "dev":
            protocol_path=os.path.join(path, "CM_protocol","cm_develop.ndx")
        elif self.part == "eval":
            protocol_path=os.path.join(path, "CM_protocol", "cm_evaluation.ndx")
        self.label_dict={"human":REAL, "spoof":FAKE}
        self.path = path
        self.protocol  = np.loadtxt(protocol_path,dtype=str)
        
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        dirname = protocol_idx[0]
        filename = protocol_idx[1]
        filepath = os.path.join(self.path,'wav',dirname,filename + ".wav")
        label = torch.tensor(self.label_dict[protocol_idx[3]], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename
    
class inthewild(ASVspoof2019LA):
    def __init__(self,path="/data/wujy/audio/inthewild/release_in_the_wild"):
        protocol_path=os.path.join(path, "meta.csv")
        self.label_dict={"bona-fide":REAL, "spoof":FAKE}
        self.path = path
        self.protocol  = np.loadtxt(protocol_path,delimiter = ",",dtype=str)[1:]
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        filename = protocol_idx[0]
        filepath = os.path.join(self.path,filename)
        label = torch.tensor(self.label_dict[protocol_idx[2]], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename
    

class FakeAVCeleb(ASVspoof2019LA):
    def __init__(self, part, path="/data/wujy/audio/fac/FakeAVCeleb_v1.2"):
        protocol_path=os.path.join(path, "meta_data.csv")
        self.label_dict={"RealVideo-RealAudio":REAL, "RealVideo-FakeAudio":FAKE,"FakeVideo-RealAudio":REAL, "FakeVideo-FakeAudio":FAKE}
        self.path = path
        self.protocol  = []
        lines=open(protocol_path,'r').readlines()
        for line in lines[1:]:
            line=line.rstrip('\n')
            line_split=line.split(",")
            self.protocol.append(line_split)
        self.protocol=np.array(self.protocol,dtype=str)
        # https://github.com/DASH-Lab/FakeAVCeleb/blob/main/train_main.py
        # val_ratio=0.3, validation ratio on trainset
        if part=='dev':
            self.protocol=self.protocol[:int(0.3*len(self.protocol))]
        elif part=='eval':
            self.protocol=self.protocol
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        dirname = protocol_idx[9].replace("FakeAVCeleb/","")
        filename = protocol_idx[8]
        mp4filepath = os.path.join(self.path, dirname,filename)
        filepath = mp4filepath.replace("mp4","wav")
        label = torch.tensor(self.label_dict[protocol_idx[5]], dtype=torch.float32)
        if not os.path.exists(filepath):
            cmd="ffmpeg -i '%s' -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav '%s'"%(mp4filepath,filepath)
            os.system(cmd)
        featureTensor = self.load_audio(filepath)
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename

class FoR(ASVspoof2019LA):
    def __init__(self,part="dev", path="/data/wujy/audio/for/for-original"):
        part=part.replace("eval","testing").replace("dev","validation")
        self.label_dict={"real":REAL, "fake":FAKE}
        self.path=os.path.join(path,part)
        self.protocol=[os.path.join(i,kk) for i,j,k in os.walk(self.path) for kk in k]
        
    
    def __getitem__(self, idx):
        protocol_idx=self.protocol[idx]
        filepath = protocol_idx
        filename = os.path.basename(filepath)
        label_=os.path.dirname(filepath).split("/")[-1]
        label = torch.tensor(self.label_dict[label_], dtype=torch.float32)
        featureTensor = self.load_audio(filepath)
        if len(featureTensor.shape)==2: # more than one channel
            featureTensor=featureTensor[0]
        featureTensor = self.pad(featureTensor)
        SpecTensor = self.extract_mel(featureTensor)
        featureTensor = torch.tensor(torch.Tensor(featureTensor), dtype=torch.float32)
        SpecTensor = torch.tensor(torch.Tensor(SpecTensor), dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, SpecTensor, label, filename


def get_dataloader(if_train=False,if_dev=False,if_eval=False,ename='',batch_size=16,num_workers=8):
    if if_train:
        dst=ASVspoof2019LA(part='train')
    elif if_dev:
        dst1=ASVspoof2019LA(part='dev')
        dst2=FoR(part='dev')
        dst3=FakeAVCeleb(part='dev')
        dst = ConcatDataset([dst1, dst2, dst3])
        del dst1,dst2,dst3
    else:
        if ename=='asvs2019la':
            dst=ASVspoof2019LA(part="eval")
        elif ename=='asvs2015e':
            dst=ASVspoof2015(part="eval")
        elif ename=='asvs2015d':
            dst=ASVspoof2015(part="dev")
        elif ename=='asvs2021df':
            dst=ASVspoof2021DF(part="eval")
        elif ename=='asvs2021la':
            dst=ASVspoof2021LA(part="eval")
        elif ename=='fac':
            dst=FakeAVCeleb(part="eval")
        elif ename=='for':
            dst=FoR(part="eval")
        elif ename=='itw':
            dst=inthewild()
    dlr=torch.utils.data.DataLoader(dst, batch_size=batch_size,num_workers=num_workers, shuffle=if_train)
    del dst
    return dlr