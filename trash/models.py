#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _make_densenet121():
    try:
        # torchvision >= 0.13
        return torchvision.models.densenet121(weights=None)
    except TypeError:
        # older torchvision
        return torchvision.models.densenet.densenet121(pretrained=False)


class _AffordanceNet(nn.Module):
    def __init__(self, use_cuda, out_channels):
        super().__init__()
        self.use_cuda = use_cuda
        self.num_rotations = 16

        self.push_color_trunk = _make_densenet121()
        self.push_depth_trunk = _make_densenet121()
        self.grasp_color_trunk = _make_densenet121()
        self.grasp_depth_trunk = _make_densenet121()

        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False)),
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=False)),
        ]))

        for name, module in self.named_modules():
            if 'push-' in name or 'grasp-' in name:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight.data)
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

        self.interm_feat = []
        self.output_prob = []

    def _forward_single_rotation(self, input_color_data, input_depth_data, rotate_idx):
        rotate_theta = np.radians(rotate_idx * (360.0 / self.num_rotations))

        affine_mat_before = np.asarray(
            [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
             [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]],
            dtype=np.float32,
        )
        affine_mat_before = torch.from_numpy(affine_mat_before).unsqueeze(0)
        if self.use_cuda:
            affine_mat_before = affine_mat_before.cuda()
        flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=False)

        color_tensor = input_color_data.cuda() if self.use_cuda else input_color_data
        depth_tensor = input_depth_data.cuda() if self.use_cuda else input_depth_data
        rotate_color = F.grid_sample(color_tensor, flow_grid_before, mode='nearest', align_corners=False)
        rotate_depth = F.grid_sample(depth_tensor, flow_grid_before, mode='nearest', align_corners=False)

        interm_push_color_feat = self.push_color_trunk.features(rotate_color)
        interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
        interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)

        interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

        affine_mat_after = np.asarray(
            [[np.cos(rotate_theta), np.sin(rotate_theta), 0],
             [-np.sin(rotate_theta), np.cos(rotate_theta), 0]],
            dtype=np.float32,
        )
        affine_mat_after = torch.from_numpy(affine_mat_after).unsqueeze(0)
        if self.use_cuda:
            affine_mat_after = affine_mat_after.cuda()
        flow_grid_after = F.affine_grid(affine_mat_after, interm_push_feat.size(), align_corners=False)

        push_out = self.pushnet(interm_push_feat)
        grasp_out = self.graspnet(interm_grasp_feat)
        push_out = F.grid_sample(push_out, flow_grid_after, mode='nearest', align_corners=False)
        grasp_out = F.grid_sample(grasp_out, flow_grid_after, mode='nearest', align_corners=False)
        push_out = F.interpolate(push_out, scale_factor=16, mode='bilinear', align_corners=False)
        grasp_out = F.interpolate(grasp_out, scale_factor=16, mode='bilinear', align_corners=False)

        return [push_out, grasp_out], [interm_push_feat, interm_grasp_feat]

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            output_prob = []
            interm_feat = []
            with torch.no_grad():
                for rotate_idx in range(self.num_rotations):
                    out_pair, feat_pair = self._forward_single_rotation(input_color_data, input_depth_data, rotate_idx)
                    output_prob.append(out_pair)
                    interm_feat.append(feat_pair)
            return output_prob, interm_feat

        if specific_rotation < 0:
            raise ValueError('specific_rotation must be provided for training forward pass')

        self.output_prob = []
        self.interm_feat = []
        out_pair, feat_pair = self._forward_single_rotation(input_color_data, input_depth_data, int(specific_rotation))
        self.output_prob.append(out_pair)
        self.interm_feat.append(feat_pair)
        return self.output_prob, self.interm_feat


class reinforcement_net(_AffordanceNet):
    def __init__(self, use_cuda):
        super().__init__(use_cuda=use_cuda, out_channels=1)


class reactive_net(_AffordanceNet):
    def __init__(self, use_cuda):
        super().__init__(use_cuda=use_cuda, out_channels=3)
