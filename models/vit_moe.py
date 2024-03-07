from functools import partial
from collections import OrderedDict
import torch.nn as nn
from einops import repeat

#from utils import cluster
#from models import se_resnext50_32x4d
from collections import OrderedDict 

import torchvision.models as models  
from models.custom_moe_layer import FMoETransformerMLP
# from .layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers  import lecun_normal_
# from ..builder import BACKBONES
import numpy as np
from collections import Counter

from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
import torch

softmax = nn.Softmax(dim=1)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 multi_tasks=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.multi_tasks = multi_tasks

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # -------------------------------------------------------------------
        mask = torch.ones_like(attn, requires_grad=False)
        if self.multi_tasks > 1:
            #print(self.multi_tasks,'ddddddddddddddddddddddd')
            for i in range(self.multi_tasks-1):
                for j in range(i+1,self.multi_tasks):
                    mask[:, :, i, j] = mask[:, :, j, i] = 0

        attn1 = attn * mask
        # attn[:,:,1,0] = attn[:,:,1,0] - 5
        # attn[:,:,1,0] = 0

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class new_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moe=False, moe_mlp_ratio=-1, moe_experts=64,
                 moe_top_k=2, moe_gate_dim=-1, world_size=1, gate_return_decoupled_activation=False,
                 moe_gate_type="noisy", vmoe_noisy_std=1, gate_task_specific_dim=-1, multi_gate=False, 
                 regu_experts_fromtask = False, num_experts_pertask = -1, num_tasks = -1,
                 gate_input_ahead = False,regu_sem=False,sem_force=False,regu_subimage=False,expert_prune=False,multi_tasks=4):
        super().__init__()
        self.moe = moe
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop, proj_drop_ratio=drop,multi_tasks=multi_tasks)
        # NOTE: drop path for stochastic depth, we shall see if 
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gate_input_ahead = gate_input_ahead
        self.expert_prune = expert_prune
        num_tasks = multi_tasks
        if moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(drop)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            moe_gate_fun = NoisyGate_VMoE

            self.mlp = FMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std, 
                                          gate_task_specific_dim=gate_task_specific_dim,multi_gate=multi_gate,
                                          regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                                          regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune)
            self.mlp_drop = nn.Dropout(drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, sem=None):
        if self.gate_input_ahead:
            gate_inp = x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if not self.moe:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp_drop(self.mlp(self.norm2(x), gate_inp, task_id, task_specific_feature, sem)))
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionTransformerMulti_MoE(nn.Module):
    def __init__(self, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.,  norm_layer=None,
                 act_layer=None,multi_tasks=1,head_idx=None,embed_mean=None,
                 moe_mlp_ratio=1,moe_experts=16,moe_top_k=4,moe_gate_dim=770,world_size=1,gate_return_decoupled_activation=False,
                 moe_gate_type="noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=False,regu_experts_fromtask = False, 
                 num_experts_pertask = -1, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                expert_prune=False, **kwargs
                 ):

        
        
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformerMulti_MoE, self).__init__()
        if isinstance(num_classes,int):
            num_classes = [num_classes] * multi_tasks

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.nocls_token = True
        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
        self.depth = depth
        self.norm_layer = norm_layer
        self.moe_experts = moe_experts
        self.moe_top_k = moe_top_k
        self.gate_return_decoupled_activation = gate_return_decoupled_activation
        self.multi_gate = multi_gate
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_ratio
        self.attn_drop_rate = attn_drop_ratio
        drop_rate = drop_ratio
        attn_drop_rate = attn_drop_ratio
        self.gate_task_specific_dim = gate_task_specific_dim
        self.gate_input_ahead = gate_input_ahead
        self.expert_prune = expert_prune
        self.num_tasks = multi_tasks
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(self.depth):
            if i % 2 == 0:
                blocks.append(Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path_ratio=dpr[i], norm_layer=self.norm_layer,multi_tasks=multi_tasks))
            else:
                blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path_ratio=dpr[i], norm_layer=norm_layer,
                              moe=True, moe_mlp_ratio=moe_mlp_ratio, moe_experts=moe_experts, moe_top_k=moe_top_k, moe_gate_dim=moe_gate_dim, world_size=world_size,
                              gate_return_decoupled_activation=self.gate_return_decoupled_activation,
                              moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std, 
                              gate_task_specific_dim=self.gate_task_specific_dim,multi_gate=self.multi_gate,
                              regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                              gate_input_ahead = self.gate_input_ahead,regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune,multi_tasks=multi_tasks))
                
                
        self.blocks = nn.Sequential(*blocks)
        
        if self.gate_task_specific_dim<0 or self.multi_gate:
            self.gate_task_represent = None
        else:
            self.gate_task_represent = new_Mlp(in_features=self.num_tasks, hidden_features=int(self.gate_task_specific_dim), out_features=self.gate_task_specific_dim,)
        
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        #self.head = nn.Linear(self.num_features, num_classes[1]) if num_classes[1] > 0 else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes[1]) if num_classes > 0 else nn.Identity()


        #-----------------------------
        self.multi_tasks = multi_tasks

        self.task_tokens = nn.Parameter(torch.zeros(1, self.multi_tasks, embed_dim))
        print(self.task_tokens.size(),self.task_tokens[:,0,:].reshape(1,1,embed_dim).size())



        self.heads = [nn.Linear(self.num_features, num_classes[i]).cuda() for i in range(self.multi_tasks) ]


        self.heads=nn.ModuleList(self.heads)
 
        self.head_idx=head_idx

        if embed_mean is None:
            self.embed_mean = torch.zeros(multi_tasks,embed_dim).numpy()
            print(self.embed_mean.shape,'embed_mean is None--------------')

        self.mu = 0.9

        # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.task_tokens, std=0.02)
        self.apply(_init_vit_weights)



    def forward_features(self, x,task_id=None,gate_inp=None,sem=None):
        # if self.task_tokens.shape[1] ==2:
        #     self.task_tokens = nn.parameter.Parameter(torch.cat([self.task_tokens,self.task_tokens[:,0,:].reshape(1,1,-1)],dim=1))

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]

        task_tokens = repeat(self.task_tokens, '() n d -> b n d', b=x.size(0))

        if self.dist_token is None:
            x = torch.cat((task_tokens, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((task_tokens, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        
        task_specific_feature = None
        if (task_id is not None) and (self.gate_task_represent is not None):
            task_specific = torch.zeros(self.num_tasks,device=x.device)
            task_specific[task_id]=1.0
            task_specific_feature = self.gate_task_represent(task_specific)
            
        for i, blk in enumerate(self.blocks):
            if blk.moe:
                x=blk(x, gate_inp, task_id, task_specific_feature,sem=sem)
            else:
                x = blk(x)
        
        
        return x[:, :self.multi_tasks,:]


    def forward(self, x,task_id=None):
        
        
        assert isinstance(task_id,(int,list,type(None))),f'need type of task_id in (int,list,None) ,but get {type(task_id)}'
        
        if isinstance(task_id,int):
            task_id = [task_id,] 
    
        x = self.forward_features(x,task_id)
        
        # print(x.sum().item(),'www')
        res = {}
        
        if task_id is None:
            task_ids = list(range(self.multi_tasks))
        else:
            task_ids = task_id
            
        # print('dsadsa',task_id)
        
        out=[]
        for i in task_ids:
            x = self.norm(x)
            pred = self.heads[i](x[:,i])
            out.append(pred)
            res[int(i)] = pred


        return res


def vit_multi_MoE(embed_dim=512, num_classes: int = 21843, has_logits: bool = True, multi_tasks=2,head_idx=None,depth=4,embed_mean=None,num_heads=16,\
                            moe_mlp_ratio=1,moe_experts=16,moe_top_k=4,moe_gate_dim=770,world_size=1,gate_return_decoupled_activation=False,
                            moe_gate_type="noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=False,regu_experts_fromtask = False, 
                            num_experts_pertask = -1, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                            expert_prune=False,drop_ratio=0.2):

    model = VisionTransformerMulti_MoE(
                            embed_dim=embed_dim,#768
                            depth=depth,
                            num_heads=num_heads,#16
                            representation_size=512 if has_logits else None,
                            num_classes=num_classes,
                            multi_tasks=multi_tasks,
                            head_idx=head_idx,
                           
                            embed_mean=None,
                            drop_ratio=drop_ratio,
                            moe_mlp_ratio=moe_mlp_ratio,moe_experts=moe_experts,moe_top_k=moe_top_k,moe_gate_dim=moe_gate_dim,world_size=world_size,gate_return_decoupled_activation=gate_return_decoupled_activation,
                            moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std, gate_task_specific_dim=gate_task_specific_dim,multi_gate=multi_gate,regu_experts_fromtask = regu_experts_fromtask, 
                            num_experts_pertask = num_experts_pertask, num_tasks = num_tasks, gate_input_ahead=gate_input_ahead, regu_sem=regu_sem, sem_force=sem_force, regu_subimage=regu_subimage, 
                            expert_prune=expert_prune, 
                                   )
    
    return model



if __name__=='__main__':
    model = VisionTransformerMulti_MoE(
                              patch_size=16,
                              embed_dim=512,
                              depth=4,
                              num_heads=2,
                              representation_size=None,
                              num_classes=2,
                              multi_tasks=2,).cuda()
    

    a=torch.FloatTensor(4,256,512).cuda()
    # model.eval()
    out=model(a)
    print(out[0].size())
    # print((out[0]==out[1]).sum()/out[1].numel())

    # print(model.heads)
    print(model.heads)