# %% [code]
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import yaml
import gc

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib
import cv2

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skimage.io import imread

# from apex.apex import amp
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.backends.cudnn as cudnn
import torchvision

import pretrainedmodels

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHES = 10
nrows = False
# %% [markdown]
# dataset and dataloader

# %% [code]
from skimage.transform import rescale


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, labels, input_w=640, input_h=512,
                 down_ratio=4, transform=None, test=False, lhalf=False,
                 hflip=0, scale=0, scale_limit=0,
                 test_img_paths=None, test_mask_paths=None, test_outputs=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.test_img_paths = test_img_paths
        self.test_mask_paths = test_mask_paths
        self.test_outputs = test_outputs
        self.input_w = input_w
        self.input_h = input_h
        self.down_ratio = down_ratio
        self.transform = transform
        self.test = test
        self.lhalf = lhalf
        self.hflip = hflip
        self.scale = scale
        self.scale_limit = scale_limit
        self.output_w = self.input_w // self.down_ratio
        self.output_h = self.input_h // self.down_ratio
        self.max_objs = 100
        self.mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)

    def __getitem__(self, index):
        img_path, mask_path, label = self.img_paths[index], self.mask_paths[index], self.labels[index]
        num_objs = len(label)

        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = cv2.resize(img, (self.input_w, self.input_h))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (self.output_w, self.output_h))
            mask = 1 - mask.astype('float32') / 255
        else:
            mask = np.ones((self.output_h, self.output_w), dtype='float32')

        if self.test:
            img = img.astype('float32') / 255
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)

            mask = mask[None, ...]

            if self.lhalf:
                img = img[:, self.input_h // 2:]
                mask = mask[:, self.output_h // 2:]

            return {
                'img_path': img_path,
                'input': img,
                'mask': mask,
            }

        kpts = []
        poses = []
        for k in range(num_objs):
            ann = label[k]
            kpts.append([ann['x'], ann['y'], ann['z']])
            poses.append([ann['yaw'], ann['pitch'], ann['roll']])
        kpts = np.array(kpts)
        poses = np.array(poses)

        if np.random.random() < self.hflip:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            kpts[:, 0] *= -1
            poses[:, [0, 2]] *= -1

        #         if np.random.random() < self.scale:
        #             scale = np.random.uniform(-self.scale_limit, self.scale_limit) + 1.0
        # #                 trans_tmp = transforms.Scale(scale)
        #             img = rescale(img,scale)
        #             mask = rescale(mask,scale)
        #             print(img.shape)
        # #                 img = F.shift_scale_rotate(img, angle=0, scale=scale, dx=0, dy=0)
        # #                 mask = F.shift_scale_rotate(mask, angle=0, scale=scale, dx=0, dy=0)
        #             kpts[:, 2] /= scale

        kpts = np.array(convert_3d_to_2d(kpts[:, 0], kpts[:, 1], kpts[:, 2])).T
        kpts[:, 0] *= self.input_w / width
        kpts[:, 1] *= self.input_h / height

        if self.transform is not None:
            data = self.transform(image=img, mask=mask, keypoints=kpts)
            img = data['image']
            mask = data['mask']
            kpts = data['keypoints']

        for k, ((x, y), (yaw, pitch, roll)) in enumerate(zip(kpts, poses)):
            label[k]['x'] = x
            label[k]['y'] = y
            label[k]['yaw'] = yaw
            label[k]['pitch'] = pitch
            label[k]['roll'] = roll

        img = img.astype('float32') / 255
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        mask = mask[None, ...]

        hm = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        reg = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        wh = np.zeros((2, self.output_h, self.output_w), dtype=np.float32)
        depth = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        eular = np.zeros((3, self.output_h, self.output_w), dtype=np.float32)
        trig = np.zeros((6, self.output_h, self.output_w), dtype=np.float32)
        quat = np.zeros((4, self.output_h, self.output_w), dtype=np.float32)
        gt = np.zeros((self.max_objs, 7), dtype=np.float32)

        for k in range(num_objs):
            ann = label[k]
            x, y = ann['x'], ann['y']
            x *= self.output_w / self.input_w
            y *= self.output_h / self.input_h
            if x < 0 or y < 0 or x > self.output_w or y > self.output_h:
                continue

            bbox = get_bbox(
                ann['yaw'],
                ann['pitch'],
                ann['roll'],
                *convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z']),
                ann['z'],
                width,
                height,
                self.output_w,
                self.output_h)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            ct = np.array([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            draw_umich_gaussian(hm[0], ct_int, radius)

            reg_mask[0, ct_int[1], ct_int[0]] = 1
            reg[:, ct_int[1], ct_int[0]] = ct - ct_int
            wh[0, ct_int[1], ct_int[0]] = w
            wh[1, ct_int[1], ct_int[0]] = h
            depth[0, ct_int[1], ct_int[0]] = ann['z']

            yaw = ann['yaw']
            pitch = ann['pitch']
            roll = ann['roll']

            eular[0, ct_int[1], ct_int[0]] = yaw
            eular[1, ct_int[1], ct_int[0]] = pitch
            eular[2, ct_int[1], ct_int[0]] = rotate(roll, np.pi)

            trig[0, ct_int[1], ct_int[0]] = math.cos(yaw)
            trig[1, ct_int[1], ct_int[0]] = math.sin(yaw)
            trig[2, ct_int[1], ct_int[0]] = math.cos(pitch)
            trig[3, ct_int[1], ct_int[0]] = math.sin(pitch)
            trig[4, ct_int[1], ct_int[0]] = math.cos(rotate(roll, np.pi))
            trig[5, ct_int[1], ct_int[0]] = math.sin(rotate(roll, np.pi))

            qx, qy, qz, qw = (R.from_euler('xyz', [yaw, pitch, roll])).as_quat()
            norm = (qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2) ** (1 / 2)
            quat[0, ct_int[1], ct_int[0]] = qx / norm
            quat[1, ct_int[1], ct_int[0]] = qy / norm
            quat[2, ct_int[1], ct_int[0]] = qz / norm
            quat[3, ct_int[1], ct_int[0]] = qw / norm

            gt[k, 0] = ann['pitch']
            gt[k, 1] = ann['yaw']
            gt[k, 2] = ann['roll']
            gt[k, 3:5] = convert_2d_to_3d(ann['x'] * width / self.input_w, ann['y'] * height / self.input_h, ann['z'])
            gt[k, 5] = ann['z']
            gt[k, 6] = 1

        if self.lhalf:
            img = img[:, self.input_h // 2:]
            mask = mask[:, self.output_h // 2:]
            hm = hm[:, self.output_h // 2:]
            reg_mask = reg_mask[:, self.output_h // 2:]
            reg = reg[:, self.output_h // 2:]
            wh = wh[:, self.output_h // 2:]
            depth = depth[:, self.output_h // 2:]
            eular = eular[:, self.output_h // 2:]
            trig = trig[:, self.output_h // 2:]
            quat = quat[:, self.output_h // 2:]

        ret = {
            'img_path': img_path,
            'input': img,
            'mask': mask,
            # 'label': label,
            'hm': hm,
            'reg_mask': reg_mask,
            'reg': reg,
            'wh': wh,
            'depth': depth,
            'eular': eular,
            'trig': trig,
            'quat': quat,
            'gt': gt,
        }
        return ret

    def __len__(self):
        if self.test_img_paths is None:
            return len(self.img_paths)
        else:
            return len(self.img_paths) + len(self.test_img_paths)


from torch.nn import Conv2d
def get_model(name, heads, head_conv=128, num_filters=[256, 256, 256],
              dcn=False, gn=False, freeze_bn=False, **kwargs):
    if 'res' in name and 'fpn' in name:
        backbone = '_'.join(name.split('_')[:-1])
        model = ResNetFPN(backbone, heads, head_conv, num_filters,
                          dcn=dcn, gn=gn, freeze_bn=freeze_bn)
    elif 'dla' in name:
        pretrained = '_'.join(name.split('_')[1:])
        model = dla.get_dla34(heads, pretrained, head_conv, num_filters,
                              gn=gn, freeze_bn=freeze_bn)
    else:
        raise NotImplementedError

    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class ResNetFPN(nn.Module):
    def __init__(self, backbone, heads, head_conv=128,
                 num_filters=[256, 256, 256], pretrained=True,
                 dcn=False, gn=False, freeze_bn=False):
        super().__init__()

        self.heads = heads

        if backbone == 'resnet18':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet34':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet101':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet152':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'se_resnext50_32x4d':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'se_resnext101_32x4d':
            pretrained = 'imagenet' if pretrained else None
            self.backbone = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet34_v1b':
            self.backbone = timm.create_model('gluon_resnet34_v1b', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50_v1d':
            self.backbone = timm.create_model('gluon_resnet50_v1d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet101_v1d':
            self.backbone = timm.create_model('gluon_resnet101_v1d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnext50_32x4d':
            self.backbone = timm.create_model('resnext50_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnext50d_32x4d':
            self.backbone = timm.create_model('resnext50d_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'seresnext26_32x4d':
            self.backbone = timm.create_model('seresnext26_32x4d', pretrained=pretrained)
            convert_to_inplace_relu(self.backbone)
            num_bottleneck_filters = 2048
        elif backbone == 'resnet18_ctdet':
            self.backbone = models.resnet18()
            state_dict = torch.load('pretrained_weights/ctdet_coco_resdcn18.pth')['state_dict']
            self.backbone.load_state_dict(state_dict, strict=False)
            num_bottleneck_filters = 512
        elif backbone == 'resnet50_maskrcnn':
            self.backbone = models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained).backbone.body
            print(self.backbone)
            num_bottleneck_filters = 2048
        else:
            raise NotImplementedError

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        self.lateral4 = nn.Sequential(
            Conv2d(num_bottleneck_filters, num_filters[0],
                   kernel_size=1, bias=False),
            nn.GroupNorm(32, num_filters[0]) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral3 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 2, num_filters[0],
                   kernel_size=1, bias=False),
            nn.GroupNorm(32, num_filters[0]) if gn else nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True))
        self.lateral2 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 4, num_filters[1],
                   kernel_size=1, bias=False),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.lateral1 = nn.Sequential(
            Conv2d(num_bottleneck_filters // 8, num_filters[2],
                   kernel_size=1, bias=False),
            nn.GroupNorm(32, num_filters) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        self.decode3 = nn.Sequential(
            DCN(num_filters[0], num_filters[1],
                kernel_size=3, padding=1, stride=1) if dcn else \
                Conv2d(num_filters[0], num_filters[1],
                       kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, num_filters[1]) if gn else nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True))
        self.decode2 = nn.Sequential(
            Conv2d(num_filters[1], num_filters[2],
                   kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))
        self.decode1 = nn.Sequential(
            Conv2d(num_filters[2], num_filters[2],
                   kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, num_filters[2]) if gn else nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True))

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                Conv2d(num_filters[2], head_conv,
                       kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, head_conv) if gn else nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        module_names = [n for n, _ in self.backbone.named_modules()]
        if 'layer0' in module_names:
            x1 = self.backbone.layer0(x)
        else:
            x1 = self.backbone.conv1(x)
            x1 = self.backbone.bn1(x1)
            x1 = self.backbone.relu(x1)
            x1 = self.backbone.maxpool(x1)
        x1 = self.backbone.layer1(x1)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        lat4 = self.lateral4(x4)
        lat3 = self.lateral3(x3)
        lat2 = self.lateral2(x2)
        lat1 = self.lateral1(x1)

        map4 = lat4
        map3 = lat3 + F.interpolate(map4, scale_factor=2, mode="nearest")
        map3 = self.decode3(map3)
        map2 = lat2 + F.interpolate(map3, scale_factor=2, mode="nearest")
        map2 = self.decode2(map2)
        map1 = lat1 + F.interpolate(map2, scale_factor=2, mode="nearest")
        map1 = self.decode1(map1)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(map1)
        return ret


heads = OrderedDict([
    ('hm', 1),
    ('reg', 2),
    ('depth', 1),
])
heads['trig'] = 6
heads['wh'] = 2


def train(heads, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter()}
    for head in heads.keys():
        avg_meters[head] = AverageMeter()

    model.train()

    pbar = tqdm(total=len(train_loader))
    for i, batch in enumerate(train_loader):
        input = batch['input'].to(device)
        mask = batch['mask'].to(device)
        reg_mask = batch['reg_mask'].to(device)

        output = model(input)

        loss = 0
        losses = {}
        for head in heads.keys():
            losses[head] = criterion[head](output[head], batch[head].to(device),
                                           mask if head == 'hm' else reg_mask)
            if head == 'wh':
                loss += 0.05 * losses[head]
            else:
                loss += losses[head]
        losses['loss'] = loss

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        #         if config['apex']:
        #             with amp.scale_loss(loss, optimizer) as scaled_loss:
        #                 scaled_loss.backward()
        #         else:
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(losses['loss'].item(), input.size(0))
        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        for head in heads.keys():
            avg_meters[head].update(losses[head].item(), input.size(0))
            postfix[head + '_loss'] = avg_meters[head].avg
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return avg_meters['loss'].avg


# %% [code]
def validate(heads, val_loader, model, criterion,ifreturndata=False):
    avg_meters = {'loss': AverageMeter()}
    for head in heads.keys():
        avg_meters[head] = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        targets = []
        preds = []

        pbar = tqdm(total=len(val_loader))
        for i, batch in enumerate(val_loader):
            input = batch['input'].to(device)
            mask = batch['mask'].to(device)
            reg_mask = batch['reg_mask'].to(device)

            output = model(input)

            # 获取label和输出值
            if ifreturndata:
                s1, s2, h, w = output['hm'].shape
                tmp = (output['hm'] * mask).cpu().detach()
                topk_scores, topk_inds = torch.topk(tmp.view(s1, s2, -1), 40)
                topk_ys = (topk_inds / w).int().float().transpose(1,2)
                topk_xs = (topk_inds % w).int().float().transpose(1,2)
                topk_zs = torch.gather(output['depth'].view(s1, s2, -1).cpu().detach(), 2, topk_inds).transpose(1,2)
                topk_trig = torch.gather(output['trig'].view(s1, 6, -1).cpu().detach(), 2, topk_inds.repeat(1, 6, 1)).transpose(1,2)
                topk_reg = torch.gather(output['reg'].view(s1, 2, -1).cpu().detach(), 2, topk_inds.repeat(1, 2, 1)).transpose(1,2)

                pred = np.zeros((s1,40,11))
                pred[:,:,0:1] = topk_xs + topk_reg[:,:,0:1]
                pred[:,:,1:2] = topk_ys + topk_reg[:,:,1:2]
                pred[:,:,2:3] = topk_zs
                pred[:,:,3:9] = topk_trig
                pred[:,:,9:10] = topk_scores.transpose(1,2)
                pred[:,:,10:11] = topk_inds.transpose(1,2)

                s1, s2, h, w = batch['hm'].shape
                tmp = (batch['hm'].cpu() * mask.cpu()).detach()
                topk_scores, topk_inds = torch.topk(tmp.view(s1, s2, -1), 40)
                topk_ys = (topk_inds / w).int().float().transpose(1, 2)
                topk_xs = (topk_inds % w).int().float().transpose(1, 2)
                topk_zs = torch.gather(batch['depth'].view(s1, s2, -1).cpu().detach(), 2, topk_inds).transpose(1, 2)
                topk_trig = torch.gather(batch['trig'].view(s1, 6, -1).cpu().detach(), 2,
                                         topk_inds.repeat(1, 6, 1)).transpose(1, 2)
                topk_reg = torch.gather(batch['reg'].view(s1, 2, -1).cpu().detach(), 2,
                                        topk_inds.repeat(1, 2, 1)).transpose(1, 2)

                target = np.zeros((s1, 40, 11))
                target[:, :, 0:1] = topk_xs + topk_reg[:, :, 0:1]
                target[:, :, 1:2] = topk_ys + topk_reg[:, :, 1:2]
                target[:, :, 2:3] = topk_zs
                target[:, :, 3:9] = topk_trig
                target[:, :, 9:10] = topk_scores.transpose(1,2)
                target[:, :, 10:11] = topk_inds.transpose(1,2)

                targets.append(target)
                preds.append(pred)
            else:
                targets = [-1]
                preds = [-1]
                confidences = [-1]

            loss = 0
            losses = {}
            for head in heads.keys():
                losses[head] = criterion[head](output[head], batch[head].cuda(),
                                               mask if head == 'hm' else reg_mask)
                if head == 'wh':
                    loss += 0.05 * losses[head]
                else:
                    loss += losses[head]
            losses['loss'] = loss

            avg_meters['loss'].update(losses['loss'].item(), input.size(0))
            postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
            for head in heads.keys():
                avg_meters[head].update(losses[head].item(), input.size(0))
                postfix[head + '_loss'] = avg_meters[head].avg
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()
    if ifreturndata:
        targets = np.concatenate(targets,axis=0)
        preds = np.concatenate(preds,axis=0)
    return avg_meters['loss'].avg,targets,preds

# %% [markdown]
# loss

# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        loss = F.binary_cross_entropy(
            input * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


class DepthL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        output = 1. / (torch.sigmoid(output) + 1e-6) - 1.
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss /= mask.sum()
        return loss


def _neg_loss(pred, gt, mask):
    pos_inds = gt.eq(1).float() * mask
    neg_inds = gt.lt(1).float() * mask

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
               neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, output, target, mask):
        output = torch.sigmoid(output)
        loss = self.neg_loss(output, target, mask)
        return loss


# %% [markdown]
# optimizor

# %% [code]
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


# %% [code]
def convert_str_to_labels(s, names=['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']):
    labels = []
    for l in np.array(s.split()).reshape([-1, 7]):
        labels.append(dict(zip(names, l.astype('float'))))
        if 'model_type' in labels[-1]:
            labels[-1]['model_type'] = int(labels[-1]['model_type'])

    return labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    x = fx * x / z + cx
    y = fy * y / z + cy

    return x, y


def convert_2d_to_3d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z

    return x, y


def get_bbox(yaw, pitch, roll, x, y, z,
             width, height, output_w, output_h,
             car_hw=1.02, car_hh=0.80, car_hl=2.31):
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(-yaw, -pitch, -roll).T
    Rt = Rt[:3, :]
    P = np.array([
        [-car_hw, 0, 0, 1],
        [car_hw, 0, 0, 1],
        [0, car_hh, 0, 1],
        [0, -car_hh, 0, 1],
        [0, 0, car_hl, 1],
        [0, 0, -car_hl, 1],
    ]).T
    P = Rt @ P
    P = P.T
    xs, ys = convert_3d_to_2d(P[:, 0], P[:, 1], P[:, 2])
    bbox = [xs.min(), ys.min(), xs.max(), ys.max()]

    bbox[0] *= output_w / width
    bbox[1] *= output_h / height
    bbox[2] *= output_w / width
    bbox[3] *= output_h / height

    return bbox


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                  [0, 1, 0],
                  [-math.sin(yaw), 0, math.cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, math.cos(pitch), -math.sin(pitch)],
                  [0, math.sin(pitch), math.cos(pitch)]])
    R = np.array([[math.cos(roll), -math.sin(roll), 0],
                  [math.sin(roll), math.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                                            bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi

    return x


# %% [code]
name = 'se_resnext50_32x4d_fpn'
arch = 'se_resnext50_32x4d_fpn'
if nrows:
    df = pd.read_csv('./data/train.csv',nrows=100)
else:
    df = pd.read_csv('./data/train.csv')
img_paths = np.array('./data/train_images/' + df['ImageId'].values + '.jpg')
mask_paths = np.array('./data/train_masks/' + df['ImageId'].values + '.jpg')
labels = np.array([convert_str_to_labels(s) for s in df['PredictionString']])

# %% [markdown]
# transform with albumentations

# %% [code]
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast

# train_transform = Compose([
#     OneOf([
#         HueSaturationValue(
#             hue_shift_limit=20,
#             sat_shift_limit=0,
#             val_shift_limit=0,
#             p=0.5,
#         ),
#         RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5,
#         ),
#     ], p=1),
# ])
train_transform = None
val_transform = None

# %% [code]
import matplotlib.pyplot as plt
from PIL import Image

folds = []
best_losses = []
kf = KFold(n_splits=5, shuffle=True, random_state=41)
EPOCHES=30

for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths)):

    best_loss = 999
    print('Fold [%d/5]' % (fold + 1))
    train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
    train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    # train
    train_set = Dataset(
        train_img_paths,
        train_mask_paths,
        train_labels,
        input_w=1280,
        input_h=1024,
        transform=train_transform,
        lhalf=True,
        hflip=0.5,
        scale=0.5,
        scale_limit=0.1,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=10,
        shuffle=True,
        num_workers=4,
    )

    val_set = Dataset(
        val_img_paths,
        val_mask_paths,
        val_labels,
        input_w=1280,
        input_h=1024,
        transform=val_transform,
        lhalf=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )
    # create model
    model = get_model(arch, heads=heads,
                      head_conv=64,
                      num_filters=[256, 128, 64],
                      dcn=False,
                      gn=False,
                      freeze_bn=False)
    model = model.to(device)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    already_load_model = False
    
    
    criterion = OrderedDict()
    for head, func in zip(['hm', 'reg', 'depth', 'trig', 'wh'], ['FocalLoss', 'L1Loss', 'L1Loss', 'L1Loss', 'L1Loss']):
        criterion[head] = eval(func)().to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = RAdam(params, lr=3e-4, weight_decay=0)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHES, eta_min=3e-5)
    # 创建记录文件
    # txtName = f"./log/log_resnet18_fold{fold}.txt"
    # f = open(txtName, "w")
    log_loss = []
    log_val_loss = []
    preds = [];
    targets = [];
    confidences = [];
    for epoch in range(EPOCHES):
        print('Epoch [{}/{}]'.format(epoch + 1, EPOCHES))
 
        # load model file
        if already_load_model == False:
            if os.path.exists('./model/model_{}_{}.pth'.format(name,fold+1)):
                model.load_state_dict(torch.load('./model/model_{}_{}.pth'.format(name,fold+1)))
                best_loss,_,_ = validate(heads,val_loader,model,criterion)
                already_load_model = True
                print('=>load saved model_{}_{}.pth'.format(name,fold+1))
        # train for one epoch
        train_loss = train(heads, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        if epoch == EPOCHES-1:
            val_loss,target,pred = validate(heads, val_loader, model, criterion,ifreturndata=True)
            targets.append(target)
            preds.append(pred)
        else:
            val_loss,_,_ = validate(heads, val_loader, model, criterion,ifreturndata=False)
        scheduler.step()
        # print('loss {} - val_loss {}'.format(train_loss, val_loss))
        # log_loss.append(str(train_loss))
        # log_val_loss.append(str(val_loss))
        #         log['epoch'].append(epoch)
        #         log['loss'].append(train_loss)
        #         log['val_loss'].append(val_loss)

        #         pd.DataFrame(log).to_csv('models/detection/%s/log_%d.csv' % (config['name'], fold+1), index=False)

        if val_loss < best_loss:
            torch.save(model.state_dict(), './model/model_{}_{}.pth'.format(name, fold + 1))
            best_loss = val_loss
            print("=> saved best model")

        state = {
            'fold': fold + 1,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, './model/checkpoint_{}.pth.tar'.format(name))
    print('val_loss: {}'.format(best_loss))
    break
preds = np.concatenate(preds,axis=0)
targets = np.concatenate(targets,axis=0)
np.save(f'./numpy_result/{name}_preds.npy',preds)
np.save(f'./numpy_result/{name}_targets.npy',targets)
    # f.write('[train_loss]\n')
    # f.write(','.join(log_loss))
    # f.write('\n[val_loss]\n')
    # f.write(','.join(log_val_loss))
