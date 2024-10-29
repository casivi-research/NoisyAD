import random
import argparse
import shutil
import time
import os
from utils.utils import find_gpus, freeze_model, read_log

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(1)

from utils.metric import *
from utils.visualizer import *

import torch
from cnn.resnet import wide_resnet50_2 as wrn50_2
from cnn.resnet import resnet18,resnet50,resnet101,resnet152,resnet34
from cnn.vgg import vgg16_bn,vgg19_bn
# from torchvision.models import vgg16_bn
from cnn.efficientnet import EfficientNet as effnet

import datasets.mvtec as mvtec
import datasets.mpdd as mpdd
from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES, MVTEC_CLASS_NAMES_1, MVTEC_CLASS_NAMES_2, MVTEC_CLASS_NAMES_3
from datasets.mpdd import MPDDDataset, MPDD_CLASS_NAMES, MPDD_CLASS_NAMES_1, MPDD_CLASS_NAMES_2, MPDD_CLASS_NAMES_3
from datasets.visa import VisADataset, VisA_CLASS_NAMES, VisA_CLASS_NAMES_1, VisA_CLASS_NAMES_2, VisA_CLASS_NAMES_3, VisA_CLASS_NAMES_4, VisA_CLASS_NAMES_5, VisA_CLASS_NAMES_6
from datasets.choose_dataset import choose_datasets

from utils.noad import *
import torch.optim as optim
import warnings
from src.src_utils import (bool_flag, initialize_exp, fix_random_seeds,
                           AverageMeter, create_logger)

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    type=str,
                    default="datasets/MVTec_AD",
                    help="path to dataset repository")
parser.add_argument("--data_type",
                    type=str,
                    default="mvtec",
                    choices=['mvtec', 'mpdd', 'visa', 'btad'])
parser.add_argument('--train_class',
                    type=str,
                    default='screw',
                    help='list of train class')
parser.add_argument('--is_train',
                    type=bool,
                    default=True,
                    help='flag on whether using normal only')
parser.add_argument('--fp_nums',
                    type=int,
                    default=56 * 56,
                    help='feature points per image')
parser.add_argument('--Rd', type=bool, default=False)
parser.add_argument("--nmb_crops",
                    type=int,
                    default=[1],
                    nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops",
                    type=int,
                    default=[224],
                    nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops",
                    type=float,
                    default=[0.14],
                    nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops",
                    type=float,
                    default=[1],
                    nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--dump_path",
                    type=str,
                    default="noisy",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--crops_for_assign",
                    type=int,
                    nargs="+",
                    default=[0],
                    help="list of crops id used for computing assignments")
parser.add_argument("--feat_dim",
                    default=1792,
                    type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes",
                    default=[10],
                    type=int,
                    nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--epochs",
                    default=100,
                    type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size",
                    default=1,
                    type=int,help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--wd", 
                    default=5e-4, 
                    type=float, 
                    help="weight decay")
parser.add_argument("--world_size",
                    default=1,
                    type=int,
                    help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank",
                    default=0,
                    type=int,
                    help="""rank of this process:
                    it is set automatically and should not be passed as argument"""
                    )
parser.add_argument("--local_rank",
                    default=0,
                    type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument("--seed", type=int, default=1024, help="seed")
args = parser.parse_args()


def main(seed, noisy_p):

    global args

    args.seed = seed
    fix_random_seeds(args.seed)
        
    args.nmb_prototypes = [10]

    args.K = 1
    args.J = 9

    args.feat_dim = 1792
    args.in_dim = 1792
    args.out_dim = 1792

    args.batch_size = 4
    args.base_lr = 1e-3
    args.wd = 5e-4
    args.Rd = False
    args.m_f = 0.8
    args.m_c = 0

    args.r = 1e-5

    args.epochs = 100
    args.checkpoint_freq = int(0.1 * args.epochs)

    NoiseList = [
        'RandomBrightnessContrast', 'GaussNoise', 'ZoomBlur', 'ISONoise',
        'Defocus'
    ]

    for noise_type in NoiseList:
        
        if args.data_type == 'mvtec':
            args.data_path = "datasets/MVTec_AD"
            args.train_class = MVTEC_CLASS_NAMES

        elif args.data_type == 'mpdd':
            args.data_path = "datasets/MPDD"
            args.train_class = MPDD_CLASS_NAMES

        elif args.data_type == 'visa':
            args.data_path = "datasets/VisA"
            args.train_class = VisA_CLASS_NAMES
            
        for train_class in args.train_class:

            # init
            args.dump_path = f"exp/{noise_type}/r={noisy_p}/{args.data_type}"
            
            args.train_class = [train_class]
            logger = initialize_exp(args, "epoch", "loss")
            shutil.copy(os.path.realpath(__file__),
                        os.path.join(args.dump_path, "snaptshot_train.py"))
            shutil.copy('utils/cad.py',
                        os.path.join(args.dump_path, "snaptshot_model.py"))

            # ============= build data ... =============
            train_dataset, test_dataset = choose_datasets(
                args, train_class, noise_type, noisy_p)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=True,
                drop_last=False,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                pin_memory=True,
            )
            logger.info("Train data with {} images loaded.".format(
                len(train_dataset)))
            logger.info("Test data with {} images loaded.".format(
                len(test_dataset)))

            # ============= build models ... =============
            # wide-resnet50
            encoder = wrn50_2(pretrained=True, progress=True) # 1792
            encoder = encoder.to(device)
            encoder.eval()

            model = NOAD(args, logger, args.nmb_prototypes[0], device, encoder, train_loader)
            model.centroids = nn.Parameter(model.centroids, requires_grad=False)

            model = model.to(device)
            logger.info("Building models done.")

            params = [
                {
                    'params': model.parameters()
                },
            ]
            optimizer = optim.AdamW(params=params,
                                    lr=args.base_lr,
                                    weight_decay=args.wd,
                                    amsgrad=True)
            logger.info("Building optimizer done.")

            # ============= start train ... =============
            for epoch in range(args.epochs):
                train_loss = AverageMeter()
                with tqdm(total=len(train_loader)) as t:

                    # train
                    model.train()
                    for it, (idx, x, _, _, _) in enumerate(train_loader):
                        optimizer.zero_grad()

                        # forward passes
                        p = encoder(x.to(device))

                        loss, _, embeds = model.forward(p)
                        loss.backward()
                        optimizer.step()

                        # flush current state
                        train_loss.update(loss.item(), x.size(0))

                        t.set_postfix({
                            'epoch':
                            f'{epoch}/{args.epochs}',
                            'class':
                            f'{train_class}',
                            'noise':
                            f'{noise_type}',
                            'loss':
                            '{:705.3f}'.format(train_loss.avg),
                        })
                        t.update()

                    if (it + 1) % len(train_loader) == 0:
                        model.eval()
                        img_roc_auc, per_pixel_rocauc, per_pixel_proauc = detection(
                            test_loader, encoder, model)
                        model.train()

                        model.best_img_roc = max(img_roc_auc,
                                                 model.best_img_roc)
                        model.best_pix_roc = max(per_pixel_rocauc,
                                                 model.best_pix_roc)
                        model.best_pix_pro = max(per_pixel_proauc,
                                                 model.best_pix_pro)

                        logger.info(f'{epoch} - {it} loss is {train_loss.avg}')
                        logger.info('%d - I_AUROC: %.5f|%.5f' %
                                    (it, img_roc_auc, model.best_img_roc))
                        logger.info('%d - P_ROCAUC: %.5f|%.5f' %
                                    (it, per_pixel_rocauc, model.best_pix_roc))
                        logger.info('%d - P_AUPRO: %.5f|%.5f' %
                                    (it, per_pixel_proauc, model.best_pix_pro))
                        logger.info(' ')

            logger.info(f'{train_class} ' + 'image ROCAUC: %.5f' %
                        (model.best_img_roc))
            logger.info(f'{train_class} ' + 'pixel ROCAUC: %.5f' %
                        (model.best_pix_roc))
            logger.info(f'{train_class} ' + 'pixel P_AUPRO: %.5f' %
                        (model.best_pix_pro))
    noise_type = None


def detection(test_loader, encoder, cad):

    # ============= start train ... =============
    gt_mask_list = list()
    gt_list = list()
    heatmaps = None
    img_roc_auc, per_pixel_rocauc, per_pixel_proauc = 0, 0, 0

    # sum_time=0
    for _, x, y, mask, _ in test_loader:
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())

        x = x.to(device)

        p = encoder(x)
        _, score, _ = cad.forward(p)
        heatmap = score.cpu().detach()
        heatmap = torch.mean(heatmap, dim=1) # 4*56*56
        heatmaps = torch.cat(
            (heatmaps, heatmap), dim=0) if heatmaps != None else heatmap # 83*56*56

    heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear') # (83,224,224), np.ndarray
    heatmaps = gaussian_smooth(heatmaps, sigma=4)

    gt_mask = np.asarray(gt_mask_list)
    scores = rescale(heatmaps)

    fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
    fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask.astype(int), scores)
    per_pixel_proauc = cal_pxl_pro(gt_mask, scores)

    return img_roc_auc, per_pixel_rocauc, per_pixel_proauc


if __name__ == '__main__':
    main(seed=0, noisy_p=0.1)
