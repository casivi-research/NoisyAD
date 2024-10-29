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
from tools.funcs import rot_img, translation_img, hflip_img, rot90_img, grey_img

from PIL import Image
import torchvision.transforms as T


class NOAD(nn.Module):

    def __init__(self, args, logger, nmb_protypes, device, encoder=None, train_loader=None):
        super(NOAD, self).__init__()

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
        self.init_features = 0

        self.r = nn.Parameter(args.r * torch.ones(1), requires_grad=True)
        self.descriptor = Descriptor(args.in_dim, args.out_dim).cuda()

        # setting for noise
        noise = torch.randn(self.args.nmb_prototypes[0], args.feat_dim)
        Q, R = torch.qr(noise.t())
        self.centroids = Q.t()

        self.best_img_roc = -1
        self.best_pix_roc = -1
        self.best_pix_pro = -1

    def forward(self, p):

        embeds = self.descriptor(p)
        embeds = rearrange(embeds, 'b c h w -> b (h w) c')

        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c
        dist = torch.sqrt(dist)

        n_neighbors = self.K
        dist = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        score = rearrange(dist, 'b (h w) c -> b c h w', h=int(math.sqrt(self.args.fp_nums)))

        loss = 0
        if self.training:
            loss = self._loss(embeds)

        return loss, score, embeds

    def _loss(self, embeds):
        features = torch.sum(torch.pow(embeds, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.centroids.t(), 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(embeds, (self.centroids.t()))
        dist = features + centers - f_c

        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, :, :self.K] - self.r**2)
        loss = (1 / self.nu) * torch.relu(score).mean()

        if self.J>0:

            score = (self.r**2 - dist[:, :, self.K:])
            # score = (self.r**2 - dist[:, :, self.K] + dist[:, :, self.K - 1])
            loss += (1 / self.nu) * torch.relu(score + self.alpha).mean()

        return loss


            
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
