import numpy as np
import torch
from torch import optim
import os



def calculate_loss(lossfn,pred,label,task_name,loss_weight,reduction,labels,cont):
    loss = lossfn(pred, label)
    error_name = []
    errors_num = 0       
    if task_name == 'label':
        if cont:
            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            if isinstance(labels,dict):
                highlabel = labels['highlabel']
                code = labels['code']
            elif isinstance(labels,list):
                highlabel = labels[-1]
                code = labels[-2]
                print(highlabel)
                print(pred_classes,label)
            error = ((highlabel.cpu() == 2) * (pred_classes.cpu() != label.cpu()))
            errors_num = error.sum().item()
            error_name = [code[i] + '\n' for i in range(error.size(0)) if error[i]]
            
    if reduction == 'none':
        if task_name == 'label':
            multilabel_msk = [ loss_weight if i == 2 else 1 if i == 1  else 0.7 for i in highlabel ] 
        else :
            multilabel_msk = [1]
        multilabel_weight = torch.Tensor(multilabel_msk).to(loss.device)
        loss = multilabel_weight * loss
        loss = loss.mean()


    return loss,errors_num,error_name


def get_optimizer(args,model):
 

    if args.backbone == 'vit_moe':
        ignored_params = list(map(id, model.model.heads.parameters()))   # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in ignored_params, model.model.parameters())
        res_params = model.resnet.parameters()
        params_total = [{'params':model.model.heads[i].parameters(), 'lr': args.lr_head[i],'momentum': 0.9, 'weight_decay': 5e-6}for i in range(args.multi_tasks) if args.lr_head[i] > 1e-8 ]
        params_total.append({'params': base_params})
        params_total.append({'params':res_params,'lr': args.lr_res,'momentum': 0.9, 'weight_decay': 5e-6})
        print(len(params_total),params_total)
        optimizer = optim.SGD( params_total, args.lr, momentum=0.9, weight_decay=5e-6)

    else:
        optimizer = optim.SGD( model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    return optimizer

def load_state(args,model):
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
    len_clks = weights_dict['model.task_tokens'].shape[1]
    if args.multi_tasks != len_clks:
        print(f'multi_tasks is {args.multi_tasks} but the cls is {len_clks}')
        for i in range(args.multi_tasks - len_clks):
            weights_dict['model.task_tokens'] = torch.cat((weights_dict['model.task_tokens'],weights_dict['model.task_tokens'][:,-1].unsqueeze(0)),dim=1) 

    print(model.load_state_dict(weights_dict, strict=False))

    del weights_dict

    return model