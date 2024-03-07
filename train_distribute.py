import os
import math
from utils.functions import get_optimizer,load_state

import torch

import torch.optim.lr_scheduler as lr_scheduler

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
from args import get_args
from my_dataset import MultiDataSetMoE

from models.vit_res_model import VitResMoE
from utils.utils import  train_one_epoch_multi,evaluate_multi
# from models.TransMIL import TransMIL

import pandas as pd
import random
import numpy as np
def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
def main(args):
    print(args.local_rank)
    print('-'*100)
    seed_reproducer(9)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


    



        # 实例化训练数据集
    train_dataset = MultiDataSetMoE(data=pd.read_csv(args.train_csv),
                              #images_class=train_images_label,
                              img_batch=args.img_batch,
                              head_idx=args.head_idx,
                              tasks=args.tasks,
                              need_patch=args.needpatch,
                              task_id=args.task_id
                                 )
    #print(train_dataset)
    # 实例化验证数据集
    val_dataset = MultiDataSetMoE(data=pd.read_csv(args.valid_csv),
                                #images_class=val_images_label,

                                img_batch=args.img_batch,
                                head_idx=args.head_idx,
                                tasks=args.tasks,
                                need_patch=args.needpatch,
                                task_id=args.task_id
                                )
    
    test_dataset = MultiDataSetMoE(data=pd.read_csv(args.test_csv),
                                #images_class=val_images_label,
                                img_batch=args.img_batch,
                                head_idx=args.head_idx,
                                tasks=args.tasks,
                                need_patch=args.needpatch,
                                task_id=args.task_id
                                )

    print(len(train_dataset),len(val_dataset),len(test_dataset))
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=0,
                                               sampler = train_sampler
                                               )
                    

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                            num_workers=0,
                                             )
                                          

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=0,)

    if args.backbone == 'vit_moe':
        model = VitResMoE().to(device)


    model = load_state(args,model)
    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))



    optimizer = get_optimizer(args,model)



    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    istrain_list = [True  if args.lr_head[i] > 1e-8 else False for i in range(args.multi_tasks)]
    print(istrain_list)


    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),device_ids=[args.local_rank], find_unused_parameters=True)



    if args.cont:
        with open(args.logdir,'w') as f:
            f.write('')
    train_error_dict = {}
    val_error_dict = {}
    test_error_dict = {}
    train_error_list = []
    test_error_list = []
    tb_writer = SummaryWriter(args.where)
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)


        train_loss, train_accs, train_error_list, train_senss, train_specs,train_aucs = train_one_epoch_multi(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    local_rank=args.local_rank,
                                    epoch=epoch,
                                multi_tasks=args.multi_tasks,
                                    istrain = istrain_list,
                                    cont=args.cont,
                                    use_weight=args.use_weight
                                    )
  
        scheduler.step()
        if args.local_rank ==0:
             print(args.local_rank,'train_acc', train_accs)
             print(args.local_rank,'train_sen', train_senss)
             print(args.local_rank,'train_spec', train_specs)
             print(args.local_rank,'train_auc', train_aucs)

           
        if epoch % 1 ==0:


            val_loss, val_accs, val_error_list, val_senss, val_specs,val_aucs = evaluate_multi(model=model,
                            data_loader=val_loader,
                            local_rank=args.local_rank,
                            epoch=epoch,
                            multi_tasks=args.multi_tasks,
                            name='val',
                            cont=True
                        )
            if args.local_rank ==0:
                print(args.local_rank,'val_acc', val_accs)
                print(args.local_rank,'val_sen', val_senss)
                print(args.local_rank,'val_spec', val_specs)
                print(args.local_rank,'val_auc', val_aucs)
        
        

            test_loss, test_accs, test_error_list, test_senss, test_specs,test_aucs = evaluate_multi(model=model,
                                data_loader=test_loader,
                                local_rank=args.local_rank,
                                epoch=epoch,
                                multi_tasks=args.multi_tasks,
                                name='test',
                                cont=True
                            )
            if args.local_rank ==0:
                print(args.local_rank,'test_acc', test_accs)
                print(args.local_rank,'test_sen', test_senss)
                print(args.local_rank,'test_spec', test_specs)
                print(args.local_rank,'test_auc', test_aucs)
        
            if args.cont and args.local_rank == 0:
                index = args.tasks.index('label') if 'label' in args.tasks else -1
                print(index,args.local_rank,'label')
                train_error_list_label = train_error_list[index]
                val_error_list = val_error_list[index]
                test_error_list = test_error_list[index]
                with open(args.logdir,'a') as f:
                    if len(train_error_list_label) > 0:
                        for i in train_error_list_label:
                            train_error_dict[i] = train_error_dict.get(i,0)+1
                        train_cont_lines = [str(k).strip() + ': ' + str(v) + '\n' for k,v in train_error_dict.items()]
                        f.write('**'*20 + f'train epoch {epoch}' + '**'*20 +'\n')
                        f.writelines(train_cont_lines)
                        
                    if len(val_error_list) > 0:
                        for i in val_error_list:
                            val_error_dict[i] = val_error_dict.get(i,0)+1
                        val_cont_lines = [str(k).strip() + ': ' + str(v) + '\n' for k,v in val_error_dict.items()]    
                        f.write(f'------------------------------eval epoch {epoch}-----------------------------------\n')
                        f.writelines(val_cont_lines)

                    if len(test_error_list) > 0:
                        for i in test_error_list:
                            test_error_dict[i] = test_error_dict.get(i,0)+1
                        test_cont_lines = [str(k).strip() + ': ' + str(v) + '\n' for k,v in test_error_dict.items()]    
                        f.write(f'------------------------------test epoch {epoch}-----------------------------------\n')
                        f.writelines(test_cont_lines)
            
                tb_writer.add_scalar(f'train_error_highpos',len(train_error_list),epoch)
                tb_writer.add_scalar(f'val_error_highpos',len(val_error_list),epoch)
                tb_writer.add_scalar(f'test_error_highpos',len(test_error_list),epoch)
            

            for i in range(args.multi_tasks):
                tb_writer.add_scalar(f'val_loss_{i}',val_loss[i],epoch)
                tb_writer.add_scalar(f'val_acc_{i}',val_accs[i],epoch)
                tb_writer.add_scalar(f'val_sen_{i}',val_senss[i],epoch)
                tb_writer.add_scalar(f'val_spec_{i}',val_specs[i],epoch)
                tb_writer.add_scalar(f'val_auc_{i}',val_aucs[i],epoch)
                
                tb_writer.add_scalar(f'test_loss_{i}',test_loss[i],epoch)
                tb_writer.add_scalar(f'test_acc_{i}',test_accs[i],epoch)
                tb_writer.add_scalar(f'test_sen_{i}',test_senss[i],epoch)
                tb_writer.add_scalar(f'test_spec_{i}',test_specs[i],epoch)
                tb_writer.add_scalar(f'test_auc_{i}',test_aucs[i],epoch)



        for i in range(args.multi_tasks):
            # ...
                tb_writer.add_scalar(f'train_loss_{i}',train_loss[i],epoch)
                tb_writer.add_scalar(f'train_acc_{i}',train_accs[i],epoch)

                
                tb_writer.add_scalar(f'train_sen_{i}',train_senss[i],epoch)
                tb_writer.add_scalar(f'train_spec_{i}',train_specs[i],epoch)             
                tb_writer.add_scalar(f'train_auc_{i}',train_aucs[i],epoch)

        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        if args.local_rank == 0:
            torch.save(model.module.state_dict(), args.where+"/model-{}-{}.pth".format(args.head_idx,epoch))
            torch.save(model.module.resnet.state_dict(),args.res_savedir+'/resnet.pth')
        torch.cuda.empty_cache()


if __name__ == '__main__':

    opt = get_args()
    opt.local_rank = int(os.environ["LOCAL_RANK"])
    print(opt)
    if opt.local_rank == 0 :
        if not os.path.exists(opt.where) :
            print(f'making dir {opt.where}')
            os.makedirs(opt.where)
        if not os.path.exists(opt.res_savedir):
            print(f'making dir {opt.res_savedir}')
            os.makedirs(opt.res_savedir)

    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(opt.local_rank)

    main(opt)
