import os
import cv2
import math
from utils import colormap
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metric import *
from utils.coordconv import Projector
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as T


class NOCO(nn.Module):

    def __init__(self, args, logger, nmb_protypes, gamma_d, device, encoder=None, train_loader=None, init_method=None):
        super(NOCO, self).__init__()

        # init
        self.device = device
        self.logger = logger
        self.args = args
        self.nu = 1e-3
        self.scale = None
        self.alpha = 1e-1
        self.K = args.K
        self.J = args.J
        self.nmb_protypes = nmb_protypes
        self.gamma_d = gamma_d
        self.init_features = 0

        self.r = nn.Parameter(args.r * torch.ones(1), requires_grad=True)
        self.descriptor = Descriptor(args.in_dim, args.out_dim)

        # setting for noiseinit
        noise = torch.randn(self.args.nmb_prototypes[0], args.feat_dim)
        Q, R = torch.qr(noise.t())
        self.centroids = Q.t()

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1
        
    def forward(self, p):

        embeds = self.descriptor(p)
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        
        embeds=embeds.reshape(-1, embeds.shape[-1])
        
        loss = 0
        if self.training:
            a_matrix=torch.matmul(embeds,embeds.t())
            loss=torch.nn.MSELoss()(a_matrix,torch.eye(a_matrix.shape[0]).cuda())

        return loss, embeds

    def ssc_forward(self, p):

        embeds = self.descriptor(p)
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        
        # 定义单位矩阵
        identity_matrix = torch.eye(embeds.size(1)).to(embeds.device)

        # 计算自相关矩阵并计算损失
        loss = 0
        for i in range(embeds.size(0)):
            X = embeds[i]  # 取出第 i 个样本，形状为 [3136, 1792]
            C = torch.matmul(X, X.T)  # 计算自相关矩阵
            diff = C - identity_matrix  # 计算差值
            loss += torch.norm(diff, p='fro')  # 计算 Frobenius 范数并累加到损失中

        # 计算平均损失
        loss /= embeds.size(0)

        return loss
    
    def ssc_score(self, p):
        embeds = self.descriptor(p)
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')
        
        # 定义单位矩阵
        identity_matrix = torch.eye(embeds.size(1)).to(embeds.device)

        # 计算自相关矩阵并计算score
        for i in range(embeds.size(0)):
            X = embeds[i]  # 取出第 i 个样本，形状为 [3136, 1792]
            C = torch.matmul(X, X.T)  # 计算自相关矩阵
            diff = C - identity_matrix  # 计算差值
            
    def ssc_score_mb(self, train_loader, encoder):
        train_feature_space=[]
        with torch.no_grad():
            for (idx, x, _, _, _) in tqdm(train_loader,
                              desc='Train set feature extracting'):
                p = encoder(x.to(self.device))
                embeds = self.descriptor(p)
                train_feature_space.append(embeds)
                
            train_feature_space = torch.cat(train_feature_space,
                                            dim=0)
            # train_feature_space = torch.cat(train_feature_space,
            #                                 dim=0).contiguous().cpu().numpy()
            return train_feature_space

class Descriptor(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Descriptor, self).__init__()
        self.layer = Projector(in_dim, out_dim, 1)\

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1)
            sample = o if sample is None else torch.cat(
                (sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)

        embeds = self.layer(sample)

        return embeds
