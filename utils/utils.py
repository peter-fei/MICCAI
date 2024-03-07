import os
import sys
import json
import pickle
import random
from utils.functions import calculate_loss

import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import random
import sys
import torchvision.models as models
from losses.losses import FocalLoss,dynamic_weight_average
from args import get_args

from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from torchmetrics import MetricCollection,Recall,Specificity,AUROC,Precision,F1Score

# from seresnext import se_resnext50_32x4d
args=get_args()
save_pth = args.res_savedir
loss_t_1 = torch.FloatTensor([0] * args.multi_tasks)
loss_t_2 = torch.FloatTensor([0] * args.multi_tasks)
loss_t_curr = torch.FloatTensor([0] * args.multi_tasks)

if args.backbone in ('vit','TransMIL'):
    resnet = models.__dict__[args.arch](pretrained=True).to('cuda')
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to('cuda')
    if os.path.isfile(args.res_weights):
        resnet.load_state_dict(torch.load(args.res_weights,map_location='cuda:0'))


# resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to('cuda')
def pre_cls_model(x):
    features = resnet(x)
    #print('sad',features.size())
    return features.squeeze()


softmax = nn.Softmax(dim=1)

device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')


def get_lossfn(name,use_reduction=False):
    if name == 'CELoss':
        return nn.CrossEntropyLoss(reduction = args.reduction if use_reduction else 'mean')
    elif name == 'FocalLoss':
        return FocalLoss(gamma=2,alpha=0.25,size_average=False)

def train_one_epoch_multi(model, optimizer, data_loader, local_rank, epoch, multi_tasks,istrain=[True]*args.multi_tasks,cont=False,use_weight=False):
    model.train()
    # model.module.resnet.eval()

    loss_functions = args.loss_fns
    accu_loss = torch.zeros(1).to(local_rank)  # 累计损失
    accu_num = torch.zeros(1).to(local_rank)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_num = [0] * multi_tasks
    accu_loss = [0] * multi_tasks
    errors_nums = [0] * multi_tasks
    errors_lists = [ [] for i in range(multi_tasks) ]
    task_epoch = [1e-8] * multi_tasks
    task_num = [1e-8] * multi_tasks
    
    sens_metrics = [ Recall(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    spec_metrics = [ Specificity(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    auc_metrics =  [ AUROC(task='multiclass',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        x = images.cuda(local_rank)
        B = x.shape[0]

        if args.backbone in ('vit','TransMIL'):
            # resnet.eval()
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            features = pre_cls_model(x)
            features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        else:#if args.backbone in ('vit_res','vit_moe','vit_moe_new'):
            features = x

        preds = model(features)

        # 
        
        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = torch.zeros(args.multi_tasks, task_id.shape[0]).cuda(local_rank)

        if args.multi_tasks > 1:
            task_id_mask[task_id,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()
        else:
            task_id_mask[0,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()
     

        code = labels['code']
        loss_total = 0
        # loss_current = []
        for i,key in enumerate(list(labels.keys())[:args.multi_tasks]):

            if not istrain[i] or task_id_mask[i].sum().item() == 0:
                continue
            pred = preds[i].cuda(local_rank)
            label = labels[key].cuda(local_rank) 
            highlabel = labels['highlabel'].cuda(local_rank)
            
            pred = pred[task_id_mask[i]==1]
            label = label[task_id_mask[i]==1]
            highlabel = highlabel[task_id_mask[i]==1]
            
            code1 = [code[k] for k in range(len(code)) if task_id_mask[i][k] == 1]
            
            loss_function = get_lossfn(loss_functions[i],use_reduction=True)

            labels1 = {'highlabel':highlabel,'code':code1,'label':label}
            loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights[i],args.reduction,labels1,cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)
             
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label.cuda(local_rank)).sum().item()

            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()

            sens_metrics[i].update(pred_classes,label)
            spec_metrics[i].update(pred_classes,label)
            auc_metrics[i].update(pred,label)

            if use_weight:  
                loss_t_2[i] = loss_t_1[i]
                loss_t_1[i] = loss.item()

    
            task_num_torch = torch.Tensor(task_num)
            if use_weight and torch.all(task_num_torch>5):
                
                weights_p = task_num_torch / task_num_torch.sum()
                weight = 1 / (weights_p[i] / weights_p).sum() + 0.25
                loss *= weight * 0.5

                
            loss_total +=loss
        
        if use_weight:
            dwa = dynamic_weight_average(loss_t_1,loss_t_2)

            loss_total += (dwa * loss_t_1).sum() * 0.5
                

        loss_noisy = collect_noisy_gating_loss(model,weight=0.01)
        loss_total+=loss_noisy
        loss_total.backward()

        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (task_epoch[i]),i,accu_num[i]/ task_num[i]) for i in args.show_tasks])
 
        s_desc = f'[train epoch {epoch}] '+ s 
        
        if cont :
            s_desc += f' error: {errors_nums[0]}'
        data_loader.desc = s_desc
        
        

        if not torch.isfinite(loss_total):
            print('WARNING: non-finite loss, ending training ', loss_total)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()


    sens_res = [sens.compute().item() if istrain[i] else 0 for i,sens in enumerate(sens_metrics)]
    spec_res = [spec.compute().item() if istrain[i] else 0 for i,spec in enumerate(spec_metrics)] 
    auc_res = [auc.compute().item() if istrain[i] else 0 for i,auc in enumerate(auc_metrics)] 
    
    if cont:

        print(f'total_errors_label : {errors_nums[0]} {len(errors_lists[0])} ')
        
    print('Train Sensitive: ',sens_res)
    print('Train Specificity: ',spec_res)
    print('Train AUC: ',auc_res)
    return np.array(accu_loss) / np.array(task_epoch), np.array(accu_num) / np.array(task_num), errors_lists, sens_res, spec_res, auc_res



@torch.no_grad()
def evaluate_multi(model, data_loader, local_rank, epoch, multi_tasks,name='valid',cont=False):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    zhenyin = 0
    gjb = 0
    sample_num = 0
    louzhen = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * multi_tasks
    accu_num = [0] * multi_tasks
    loss_functions = args.loss_fns
    errors_nums = [0] * multi_tasks
    errors_lists = [ [] for i in range(multi_tasks) ]
    task_epoch = [1e-8] * multi_tasks
    task_num = [1e-8] * multi_tasks
    
    sens_metrics = [ Recall(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    spec_metrics = [ Specificity(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    auc_metrics =  [ AUROC(task='multiclass' ,num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    pre_metrics = [ Precision(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    f1_metrics = [ F1Score(task='multiclass' if i>2 else 'binary',num_classes=i,average='macro').cuda(local_rank) for i in args.num_classes ]
    
    for step, data in enumerate(data_loader):
        
        images, labels = data
        sample_num += images.shape[0]
        
        x = images.cuda(local_rank)
        
        B = x.shape[0]
        
        if args.backbone in ('vit','TransMIL'):
            resnet.eval()
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            features = pre_cls_model(x)
            features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        else:#if args.backbone in ('vit_res','vit_moe','vit_moe_new'):
            features = x
 

        preds = model(features)
        code = labels['code']
        loss_total = 0

        task_id = labels['task_id'].cuda(local_rank)
        task_id_mask = torch.zeros(args.multi_tasks, task_id.shape[0]).cuda(local_rank)
        if args.multi_tasks > 1:
            task_id_mask[task_id,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()
        else:
            task_id_mask[0,range(task_id.shape[0])]=1
            task_id_mask = task_id_mask.long()

        for i,key in enumerate(list(labels.keys())[:multi_tasks]):

            if task_id_mask[i].sum().item() == 0:
                continue
            pred = preds[i].cuda(local_rank)
            label = labels[key].cuda(local_rank) 
            highlabel = labels['highlabel'].cuda(local_rank)
            
            pred = pred[task_id_mask[i]==1]
            label = label[task_id_mask[i]==1]
            highlabel = highlabel[task_id_mask[i]==1]
            
            code1 = [code[k] for k in range(len(code)) if task_id_mask[i][k] == 1]
    
            labels1 = {'highlabel':highlabel,'code':code1,'label':label}
            loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights[i],args.reduction,labels1,cont)
            accu_loss[i] += loss.detach().item()
            errors_nums[i] += errors_num
            errors_lists[i].extend(error_name)
            
            loss_function = get_lossfn(loss_functions[i],use_reduction=False)
            loss = loss_function(pred,label)
            if args.reduction == 'none':
                loss = loss.mean()
            
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label).sum().item()
            
            loss_total +=loss
            if args.num_classes[i] > 2:
                pred_classes = torch.softmax(pred, dim=1)
            sens_metrics[i].update(pred_classes,label)
            spec_metrics[i].update(pred_classes,label)
            auc_metrics[i].update(pred,label)
            pre_metrics[i].update(pred_classes,label)
            f1_metrics[i].update(pred_classes,label)

            task_epoch[i] += 1
            task_num[i] += task_id_mask[i].sum().item()
            
        loss_noisy = collect_noisy_gating_loss(model,weight=0.01)
        loss_total+=loss_noisy
        
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (task_epoch[i]),i,accu_num[i]/ task_num[i]) for i in args.show_tasks])
        s_desc = f'[{name} epoch {epoch}] '+ s
        
        if cont :
            s_desc += f' error: {errors_nums[0]}'

        data_loader.desc = s_desc

    sens_res = [sens.compute().item()  for i,sens in enumerate(sens_metrics)  ]
    spec_res = [spec.compute().item() for i,spec in enumerate(spec_metrics) ] 
    auc_res = []
    for i,auc in enumerate(auc_metrics):
        try:
            auc_res.append(auc.compute().item())
        except:
            auc_res.append(0)
    pre_res = [pre.compute().item() for i,pre in enumerate(pre_metrics) ] 
    f1_res = [f1.compute().item() for i,f1 in enumerate(f1_metrics) ] 

    if cont:

        print(f'total_errors_label : {errors_nums[0]} {len(errors_lists[0])} ')
        
    print(f'{name} Senstive: ',sens_res)
    print(f'{name} Specificity: ',spec_res)
    print(f'{name} AUC: ',auc_res)
    print(f'{name} Precision: ',pre_res)
    print(f'{name} F1_Score: ',f1_res)
    return np.array(accu_loss) / np.array(task_epoch), np.array(accu_num) / np.array(task_num), errors_lists, sens_res, spec_res, auc_res



def collect_noisy_gating_loss(model, weight):
    loss = 0
    for module in model.modules():
        if  isinstance(module, NoisyGate_VMoE) and module.has_loss:

            loss += module.get_loss()
    return loss * weight