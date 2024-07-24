import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
torch.set_default_tensor_type(torch.FloatTensor)
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve,confusion_matrix,auc,roc_curve
from models.AMSDF import Module as AMSDF
from data_io import get_dataloader
from tools.startup_config import set_random_seed
from tools.tool_eval import compute_eer
import warnings
warnings.filterwarnings("ignore")
__author__ = "JunyanWu"
__email__ = "wujy298@mail2.sysu.edu.cn"


def produce_evaluation_file(eval_loader, model, save_path=False):
    model.eval()
    with torch.no_grad():
        fname_list = []
        score_list = []
        y_list=[]
        for batch_x, batch_emo, batch_y, filenames in tqdm(eval_loader, ncols=50):
            batch_y = batch_y.view(-1).long()
            batch_x, batch_emo, batch_y = batch_x.float().cuda(),batch_emo.float().cuda(), batch_y.cuda()
            batch_out = model(batch_x, batch_emo, Freq_aug=False)
            fname_list.extend(filenames)
            score_list.extend(batch_out[:,1].data.cpu().numpy().ravel().tolist())
            y_list.extend(batch_y.data.cpu().numpy().ravel())
        if save_path:
            with open(save_path, 'w') as fh:
                for fn,score,label in zip(fname_list,score_list,y_list):
                    fh.write('{} {} {} \n'.format(fn,score,label))
            print('Scores saved to {}'.format(save_path))
        label_np = np.array(y_list).reshape(-1,1)
        eer=compute_eer(label_np, np.array(score_list, dtype=float))
    return eer

def test_epoch(dev_loader, model):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        num_total = 0.0
        weight = torch.FloatTensor([0.1, 0.9]).float().cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
        score_list=[]
        y_list=[]
        for batch_x, batch_emo, batch_y,_ in tqdm(dev_loader,ncols=50):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_y = batch_y.view(-1).long()
            batch_x, batch_emo, batch_y = batch_x.float().cuda(),batch_emo.float().cuda(), batch_y.cuda()
            batch_out = model(batch_x,batch_emo, Freq_aug=False)
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            score_list.extend(batch_out[:,1].data.cpu().numpy().ravel().tolist())
            y_list.extend(batch_y.data.cpu().numpy().ravel())
        label_np = np.array(y_list).reshape(-1,1)
        eer=compute_eer(label_np, np.array(score_list, dtype=float))
        val_loss /= num_total
    return val_loss,eer

def train_epoch(train_loader, model,optim):
    running_loss = 0
    num_total = 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).float().cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_emo, batch_y,_ in tqdm(train_loader,ncols=50):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_y = batch_y.view(-1).long()
        batch_x, batch_emo, batch_y = batch_x.float().cuda(),batch_emo.float().cuda(), batch_y.cuda()
        batch_out = model(batch_x, batch_emo, Freq_aug=True)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss /= num_total
    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AMSDF')
    parser.add_argument('--mn', type=str, default="AMSDF",help="model name")
    parser.add_argument('--modelroot', type=str, default="./checkpoints",help="/path/to/save/models")
    """For training"""
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-e','--num_epoch', type=int, default=15) #15~20
    parser.add_argument('-we','--warm_epoch', type=int, default=3)
    parser.add_argument('-se','--stop_epoch', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=10)#10~16
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--save', action='store_true', default=False)
    """For testing"""
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--emodel', type=str, default="./checkpoints/M1.pth",help="/path/to/load/pre-trained/models")
    parser.add_argument('--ename', type=str, default="itw")
    parser.add_argument('--score_path', type=str, default="./scores")
    args = parser.parse_args()
    set_random_seed(args.seed)
    assert args.ename in ['asvs2019la', 'asvs2021la','asvs2021df','asvs2015d','asvs2015e','for','fac','itw']
    model = AMSDF().cuda().float() 
    """Testing"""
    if args.eval:
        print('Loading checkpoint from {}'.format(args.emodel))
        model.load_state_dict(torch.load(args.emodel), False)
        score_file=os.path.join(args.score_path,'%s'%os.path.basename(args.emodel).replace(".pth",""),"%s.txt"%(args.ename))
        os.makedirs(os.path.dirname(score_file),exist_ok=True)
        eval_dlr = get_dataloader(batch_size=args.test_batch_size,num_workers=args.num_workers, if_eval=True, ename=args.ename)
        eval_eer=produce_evaluation_file(eval_dlr, model, save_path=score_file)
        print('Evaluating {} dataset, EER={:.4f}'.format(args.ename, eval_eer))
        sys.exit(0)
    """Training"""
    model_tag = 'seed{}_loss{}_lr{:7f}_wd{}_bs{}'.format(args.seed,args.loss,args.lr, args.weight_decay, args.batch_size)
    model_save_path = os.path.join(args.modelroot, args.mn, model_tag)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if not os.path.exists(model_save_path) and args.save:
        os.makedirs(model_save_path, exist_ok=True)
    train_dlr = get_dataloader(batch_size=args.batch_size,num_workers=args.num_workers, if_train=True)
    dev_dlr = get_dataloader(batch_size=args.test_batch_size,num_workers=args.num_workers, if_dev=True)
    best_dev_eer=999
    stop=0
    for epoch in range(1, args.num_epoch+1):
        print('---------',epoch,'---------')
        if stop>=args.stop_epoch:
            print('Early Stop.')
            sys.exit(0)
        train_loss = train_epoch(train_dlr, model, optimizer)
        if epoch<=args.warm_epoch:
            print('train_loss{:.4f}'.format(train_loss))
            continue
        dev_loss,dev_eer= test_epoch(dev_dlr, model)
        print('train_loss{:.4f}\tdev_loss{:.4f}\tdev_deer{:.4f}'.format(train_loss,dev_loss,dev_eer))
        if dev_eer>best_dev_eer:
            stop+=1
            continue
        best_dev_eer, stop = dev_eer, 0
        if args.save:
            modelfile=os.path.join(model_save_path,'e{}_tloss{:.4f}_dloss{:.4f}_deer{:.4f}.pth'.format(epoch,train_loss,dev_loss,dev_eer))
            torch.save(model.state_dict(), modelfile)



