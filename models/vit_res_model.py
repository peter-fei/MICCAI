import torch
import torch.nn as nn
from models.vit_moe import vit_multi_MoE

from torchvision import models
from args import get_args
import os

args = get_args()



class VitResMoE(nn.Module):
    def __init__(self,embed_dim=512,num_classes=args.num_classes, has_logits=False,multi_tasks=args.multi_tasks,\
                depth=args.depth,moe_top_k=args.moe_top_k,moe_gate_dim=args.gate_dim,moe_experts=args.moe_experts,\
                noisy_std=args.noisy_std):
        super().__init__()
        resnet = models.__dict__[args.arch](pretrained=True)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet = resnet
        embed_dim = list(dict(resnet.named_parameters()).values())[-1].shape[-1]
        print(f'embed_dim is {embed_dim}----------------------------------------------------------------')
        
        if args.res_weights != '' and os.path.isfile(args.res_weights):
            print('load resnet weights -------------------------------------------------',self.resnet.load_state_dict(torch.load(args.res_weights,map_location='cuda:0')))
        
        if args.gate_dim is None:
            gate_dim = embed_dim
        else:
            gate_dim = args.gate_dim
        
        self.model = vit_multi_MoE(embed_dim=embed_dim, num_classes= num_classes, has_logits = has_logits, multi_tasks=multi_tasks,depth=depth,num_heads=16,\
                            moe_mlp_ratio=1,moe_experts=moe_experts,moe_top_k=moe_top_k,moe_gate_dim=gate_dim,world_size=1,gate_return_decoupled_activation=False,
                            moe_gate_type="noisy_vmoe", vmoe_noisy_std=noisy_std, gate_task_specific_dim=-1,multi_gate=True,regu_experts_fromtask = False, 
                            num_experts_pertask = 1, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                            expert_prune=False)


    def forward(self, x,task_id=None):
        B = x.shape[0]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        features = self.resnet(x).squeeze()

        features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        
        preds = self.model(features,task_id)
        return preds


if __name__ == '__main__':
    net = VitResMoE().to('cuda')
    print(net)
