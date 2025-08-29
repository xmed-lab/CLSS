import torch
from torch import Tensor
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from torch.nn import functional as F
import numpy as np
import random
# from numpy import dot, argsort
# from numpy import sign, count_nonzero, ones, shape, reshape, eye, dot, argsort
# from numpy.linalg import eig, eigh


from torch import dot, argsort
from torch import sign, count_nonzero, ones, reshape, eye, dot, argsort
from torch.linalg import eig, eigh

from scipy.stats import kendalltau
from torchmetrics.regression import KendallRankCorrCoef

from matplotlib import pyplot as plt


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


kendalrankloss = KendallRankCorrCoef()

def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None





def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return np.eye(n) - np.ones((n, n)) / n


def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = np.linalg.eigh(H)
    ind = np.argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]  # second axis !!
    # if (verbose):
    #     print("...forming the Z-basis")
    #     print("check eigenvalues: ", allclose(
    #         s, concatenate((ones(n - 1), [0]), 0)))

    Z = Zp[:, :(n - 1)]
    # if (verbose):
    #     print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
    #     print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n - 1)))
    return Z


def compute_upsets(r, C, verbose=True, which_method=""):
    n = r.shape[0]
    totmatches = count_nonzero(C) / 2
    if (len(r.shape) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1)).cuda()
    # Chat = r.dot(e.T) - e.dot(r.T)
    Chat = torch.matmul(r, e.T) - torch.matmul(e, r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2 * (upsetsplus < upsetsminus) - 1
    # if (verbose):
    #     print(which_method + " upsets(+): %.4f" %
    #           (upsetsplus / float(2 * totmatches)))
    #     print(which_method + " upsets(-): %.4f" %
    #           (upsetsminus / float(2 * totmatches)))
    return upsetsplus / float(2 * totmatches), upsetsminus / float(2 * totmatches), winsign

def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = torch.diag(G.sum(dim=1))
    L = D - G

    return L


def get_ulbps(simMat, lb_val):

    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one 
    S = simMat

    n = S.shape[0]
    Z = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().cuda()

    # print(S.shape)
    Ls = GraphLaplacian(S)
    ztLsz = torch.matmul(torch.matmul(Z.T, Ls), Z)
    w, v = eig(ztLsz)
    w = torch.view_as_real(w)[:,0]
    v = torch.view_as_real(v)[...,0]

    if torch.is_complex(w):
        print("complex")
        return lb_val, False

    ind = torch.argsort(w)
    v = v[:, ind]
    r = reshape(torch.matmul(Z,v[:, 0]), (n, 1))

    _, _, rsign = compute_upsets(r, S, verbose=False)

    r_final = rsign * r
    ### r_final is shape [n, 1]
    r_rank = torch.argsort(torch.argsort(r_final.reshape(-1)))
    # print(r_final)
    r_rank_lb = r_rank[:n//2]
    r_rank_ulb = r_rank[n//2:]

    
    # print(r_rank_lb)
    # print(r_rank_ulb)

    r_rank_ulb_diff = torch.abs(r_rank_ulb.reshape(-1, 1) - r_rank_lb.reshape(-1, 1).T)
    # print(r_rank_ulb_diff)
    # print(r_rank_ulb_diff.shape)
    # print(r_rank_ulb_diff[1])
    # r_rank_ulb_diff_minindx = torch.argmin(r_rank_ulb_diff, dim = 1)
    # r_rank_ulb_diff_nxtlrg = r_rank_ulb_diff.clone()
    # print(r_rank_ulb_diff_nxtlrg)
    # print(r_rank_ulb_diff_minindx)
    # r_rank_ulb_diff_nxtlrg[:,r_rank_ulb_diff_minindx] = 9999
    # print(r_rank_ulb_diff_nxtlrg)

    r_rank_ulb_diff_minindx_btm2 = torch.topk(r_rank_ulb_diff, 2, dim = 1, largest = False)[1]
    r_rank_ulb_diff_minvalu_btm2 = torch.topk(r_rank_ulb_diff, 2, dim = 1, largest = False)[0]

    # print(torch.topk(r_rank_ulb_diff, 2, dim = 1, largest = False)[1])
    # exit()
    r_rank_ulb_diff_nxtlrg_min1 = r_rank_ulb_diff_minindx_btm2[:,0]
    r_rank_ulb_diff_nxtlrg_min2 = r_rank_ulb_diff_minindx_btm2[:,1]
    
    r_rank_ulb_diff_nxtlrg_min1_val = r_rank_ulb_diff_minvalu_btm2[:,0]
    r_rank_ulb_diff_nxtlrg_min2_val = r_rank_ulb_diff_minvalu_btm2[:,1]

    # print(r_rank_ulb_diff_minindx)
    # print(r_rank_ulb_diff_minindx.shape)
    # print("NEED TO DOUBLE CHECK DIMENSIONS")
    # print(lb_val.shape)
    r_rank_ulb_ps_sml = torch.gather(lb_val, 0, r_rank_ulb_diff_nxtlrg_min1.reshape(-1, 1))
    r_rank_ulb_ps_lrg = torch.gather(lb_val, 0, r_rank_ulb_diff_nxtlrg_min2.reshape(-1, 1))
    # return torch.cat((lb_val,r_rank_ulb_ps),dim=0 )#, True

    r_rank_ulb_ps_wght = (r_rank_ulb_diff_nxtlrg_min2_val.reshape(-1, 1) * r_rank_ulb_ps_sml + r_rank_ulb_diff_nxtlrg_min1_val.reshape(-1, 1) * r_rank_ulb_ps_lrg) / (r_rank_ulb_diff_nxtlrg_min1_val.reshape(-1, 1) + r_rank_ulb_diff_nxtlrg_min2_val.reshape(-1, 1))
    
    
    # print(r_rank_ulb_ps_wght.shape, r_rank_ulb_diff_nxtlrg_min2_val.shape, r_rank_ulb_ps_sml.shape, (r_rank_ulb_diff_nxtlrg_min2_val * r_rank_ulb_ps_sml).shape) 
    # return torch.cat((lb_val,(r_rank_ulb_ps_sml + r_rank_ulb_ps_lrg) / 2),dim=0 ), r_rank
    return torch.cat((lb_val,r_rank_ulb_ps_wght),dim=0 ), r_rank
    # return torch.cat((lb_val,r_rank_ulb_ps_sml),dim=0 ), r_rank


    # print(r_final)
    # print(kendalltau(r_final, np.array([i for i in range(n)])))


def get_ulbps_ulbonly(simMat):

    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one 
    S = simMat

    n = S.shape[0]
    Z = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().cuda()

    # print(S.shape)
    Ls = GraphLaplacian(S)
    ztLsz = torch.matmul(torch.matmul(Z.T, Ls), Z)
    w, v = eig(ztLsz)
    w = torch.view_as_real(w)[:,0]
    v = torch.view_as_real(v)[...,0]

    if torch.is_complex(w):
        print("complex")
        return None

    ind = torch.argsort(w)
    v = v[:, ind]
    r = reshape(torch.matmul(Z,v[:, 0]), (n, 1))

    _, _, rsign = compute_upsets(r, S, verbose=False)

    r_final = rsign * r
    ### r_final is shape [n, 1]
    r_rank = torch.argsort(torch.argsort(r_final.reshape(-1)))
    
    return r_rank




def get_ulbps_valtau(simMat, lb_val):

    #### input is (lb, unlb) X (lb, unlb) sim matrix
    #### output is (lb + ulb_pslb_tp), ### keep simple for now, just take the closes one 
    S = simMat

    n = S.shape[0]
    Z = torch.tensor(get_the_subspace_basis(n, verbose=False)).float().cuda()

    # print(S.shape)
    Ls = GraphLaplacian(S)
    ztLsz = torch.matmul(torch.matmul(Z.T, Ls), Z)
    w, v = eig(ztLsz)
    w = torch.view_as_real(w)[:,0]
    v = torch.view_as_real(v)[...,0]

    if torch.is_complex(w):
        print("complex")
        return lb_val, False

    ind = torch.argsort(w)
    v = v[:, ind]
    r = reshape(torch.matmul(Z,v[:, 0]), (n, 1))

    _, _, rsign = compute_upsets(r, S, verbose=False)

    r_final = rsign * r
    ### r_final is shape [n, 1]
    # print(r_final.shape, lb_val.shape)
    # print(torch.abs(kendalrankloss(r_final.reshape(-1), lb_val.reshape(-1))), torch.abs(kendalrankloss(torch.argsort(torch.argsort(r_final.reshape(-1))), lb_val.reshape(-1))))
    return torch.abs(kendalrankloss(r_final.reshape(-1), lb_val.reshape(-1)))


    # print(r_final)
    # print(kendalltau(r_final, np.array([i for i in range(n)])))











def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ResNet_CTR(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        relu_TF = True
    ) -> None:
        super(ResNet_CTR, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # if relu_TF:
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128))
        # else:            
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 128))
        #### CHANGED 04/09
        if relu_TF:
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128))
        else:            
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 64))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_common = torch.flatten(x, 1)
        x_ctrst = self.fc_ctr(x_common)
        # x_ctrst = F.normalize(x_ctrst, dim=1)


        x_reg = self.fc(x_common)

        return x_reg, x_ctrst

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)






class ResNet_CTRNP(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        relu_TF = True
    ) -> None:
        super(ResNet_CTRNP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # if relu_TF:
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128))
        # else:            
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 128))
        #### CHANGED 04/09
        if relu_TF:
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 64))
        else:            
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 64))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_common = torch.flatten(x, 1)
        # x_ctrst = self.fc_ctr(x_common)
        # x_ctrst = F.normalize(x_common, dim=1)


        x_reg = self.fc(x_common)

        return x_reg, x_common

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)












class ResNet_UCVME(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        relu_TF = True,
        drp_p = 0.2
    ) -> None:
        super(ResNet_UCVME, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_var = nn.Linear(512 * block.expansion, 1)

        
        self.drop_rate = drp_p

        # if relu_TF:
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 128))
        # else:            
        #     self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 128))
        #### CHANGED 04/09
        if relu_TF:
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 512 * block.expansion), nn.ReLU(), nn.Linear(512 * block.expansion, 64))
        else:            
            self.fc_ctr = nn.Sequential(nn.Linear(512 * block.expansion, 64))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = nn.functional.dropout(x, p=self.drop_rate, training=True)
        x = self.layer2(x)
        x = nn.functional.dropout(x, p=self.drop_rate, training=True)
        x = self.layer3(x)
        x = nn.functional.dropout(x, p=self.drop_rate, training=True)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_common = torch.flatten(x, 1)
        x_common = nn.functional.dropout(x_common, p=self.drop_rate, training=True)
        # x_ctrst = self.fc_ctr(x_common)
        # x_ctrst = F.normalize(x_common, dim=1)


        x_reg = self.fc(x_common)
        x_var = self.fc_var(x_common)

        return x_reg, x_var

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


























def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model






def _resnet_CTR(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_CTR:
    model = ResNet_CTR(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)
    return model



def _resnet_CTRNP(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_CTRNP:
    model = ResNet_CTRNP(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)
    return model


def _resnet_UCVME(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_UCVME:
    model = ResNet_UCVME(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict = False)
    return model









def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50_CTR(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_CTR('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)





def resnet50_CTRNP(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet_CTRNP:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_CTRNP('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




def resnet50_UCVME(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet_UCVME:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_UCVME('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)





class SupConLoss_admargin(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, dist = None, norm_val = 0.2, scale_s = 150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            # print(anchor_count)
            # exit()
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print(dist)
        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss 




class SupConLoss_admargin_semi(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_semi, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels_raw=None, mask=None, dist_raw = None, norm_val = 0.2, scale_s = 150, unlb_ref = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        dist_raw_expand = dist_raw.unsqueeze(dim=-1)
        dist_raw_expand = dist_raw_expand.expand(-1, dist_raw_expand.shape[0])
        dist_raw_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_raw_expand, dist_raw_expand.T)), norm_val),0,2) - 1


        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3])/4

        # print(dist_raw_abdiff.shape)
        # print(feat_cos_lblb0.shape)
        # exit()
        feat_cos_lbub1 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*3:lbulb_bsize*4])/4
        feat_cos_lbulb2 = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_ulbulb = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4])/4

        feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)
        # feat_lbulb_cos = torch.cat((torch.cat((dist_raw_abdiff, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        dist, labels_ulpbs = get_ulbps(feat_lbulb_cos, labels_raw)
        labels = labels_ulpbs.contiguous().view(-1, 1)
        # print("dist.shape", dist.shape)
        dist = dist.reshape(-1)
        # print("labels.shape", labels.shape)
        batch_size = labels.shape[0]
        # print(batch_size)

        lb_ulb = torch.cat((labels_raw.reshape(-1), unlb_ref.reshape(-1)), dim = 0)
        # print(labels_ulpbs)
        # print(lb_ulb)
        ktau = torch.abs(kendalrankloss(labels_ulpbs, lb_ulb))
        ktau_dist = torch.abs(kendalrankloss(dist, lb_ulb))

        #### debug code
        '''
        plt.scatter(lb_ulb[lb_ulb.shape[0]//2:].cpu().numpy(), dist[lb_ulb.shape[0]//2:].cpu().numpy())
        plt.scatter(lb_ulb[:lb_ulb.shape[0]//2].cpu().numpy(), dist[:lb_ulb.shape[0]//2].cpu().numpy())
        plt.savefig("/home/wdaiaj/projects/ssl_contrast/debug/pseudolabplot.png")
        plt.close()

        plt.scatter(torch.argsort(torch.argsort(lb_ulb[lb_ulb.shape[0]//2:])).cpu().numpy(), torch.argsort(torch.argsort(dist[lb_ulb.shape[0]//2:])).cpu().numpy())
        

        ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulpbs[lb_ulb.shape[0]//2:]))
        ulb_ranked = torch.argsort(torch.argsort(lb_ulb[lb_ulb.shape[0]//2:]))

        ps_ulb_ranked_ord = ps_ulb_ranked[torch.argsort(ps_ulb_ranked)]        
        ulb_ranked_ord = ulb_ranked[torch.argsort(ps_ulb_ranked)]
        
        if kendalrankloss(dist, lb_ulb) / torch.abs(kendalrankloss(dist, lb_ulb)) < 0:
            plt.scatter(ulb_ranked_ord.cpu().numpy(), ps_ulb_ranked_ord.cpu().numpy())
        else:
            plt.scatter(ulb_ranked_ord.cpu().numpy(), len(ps_ulb_ranked_ord)-1 - ps_ulb_ranked_ord.cpu().numpy())

        ps_ulb_ranked_ord_exp = ps_ulb_ranked_ord.unsqueeze(dim=-1)
        ps_ulb_ranked_ord_exp = ps_ulb_ranked_ord_exp.expand(-1, ps_ulb_ranked_ord.shape[0])
        ps_ulb_ranked_ord_diff = ps_ulb_ranked_ord_exp - ps_ulb_ranked_ord_exp.T

        ulb_ranked_ord_exp = ulb_ranked_ord.unsqueeze(dim=-1)
        ulb_ranked_ord_exp = ulb_ranked_ord_exp.expand(-1, ps_ulb_ranked_ord.shape[0])
        ulb_ranked_ord_diff = ulb_ranked_ord_exp - ulb_ranked_ord_exp.T

        ulb_ranked_ord_diff_abs = torch.abs(ulb_ranked_ord_diff)
        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ranked_ulblbdiff.csv', ulb_ranked_ord_diff_abs.cpu().numpy(), delimiter=',')

        # feat_cos_ulbulb_np = feat_cos_ulbulb.cpu().numpy()
        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ulb_lb_sim.csv', feat_cos_ulbulb.detach().cpu().numpy(), delimiter=',')

        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ulb_lb_sim_0.csv', feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2].detach().cpu().numpy(), delimiter=',')
        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ulb_lb_sim_1.csv', feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2].detach().cpu().numpy(), delimiter=',')
        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ulb_lb_sim_2.csv', feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4].detach().cpu().numpy(), delimiter=',')
        np.savetxt('/home/wdaiaj/projects/ssl_contrast/debug/ulb_lb_sim_3.csv', feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2].detach().cpu().numpy(), delimiter=',')




        ### base closest 
        print(ps_ulb_ranked_ord_diff)
        print(ulb_ranked_ord_diff)

        plt.savefig("/home/wdaiaj/projects/ssl_contrast/debug/pseudolabplot_rank.png")
        plt.close()
        exit()
        
        '''
        # print(lb_ulb)
        # print(dist.int())
        # print(lb_ulb - dist)
        # if not_complex:      
        feat_cosim_ctr = feat_cosim      
        # else:
        #     feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], \
        #                                             feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), \
        #                                 torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], \
        #                                             feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)


        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss , ktau, ktau_dist




class SupConLoss_admargin_semi_lbulbsplit(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_semi_lbulbsplit, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels_raw=None, mask=None, dist_raw = None, norm_val = 0.2, scale_s = 150, unlb_ref = None, threshold = -1, base_num = -1, hg_margin = 0.2):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        dist_raw_expand = dist_raw.unsqueeze(dim=-1)
        dist_raw_expand = dist_raw_expand.expand(-1, dist_raw_expand.shape[0])

        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_lbub1 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*3:lbulb_bsize*4])/4
        feat_cos_lbulb2 = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_ulbulb = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4])/4


        #### Compute using labeled and ulabled
        # feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        # dist_ulbpbs, labels_ulpbs = get_ulbps(feat_lbulb_cos, labels_raw)
        # labels_ulbpsdornk = labels_ulpbs[labels_ulpbs.shape[0]//2:].view(-1)
        # dist_ulbpsdornk = dist_ulbpbs[dist_ulbpbs.shape[0]//2:].view(-1)        

        # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        # ktau_dist = torch.abs(kendalrankloss(dist_ulbpsdornk, unlb_ref))


        #### Compute using unlabeled only 
        labels_ulpbs = get_ulbps_ulbonly(feat_cos_ulbulb)
        labels_ulbpsdornk = labels_ulpbs    

        ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        ktau_dist = ktau




        labels = labels_raw.contiguous().view(-1, 1)
        dist = dist_raw
        batch_size = labels.shape[0]
        
  
        #### We take labeled only
        feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), 
                                    torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)      

        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        ##### do hinge loss for unlb samples
        ulb0ulb0sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2]
        ulb0ulb1sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4]
        ulb1ulb0sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2]
        ulb1ulb1sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4]

        ### try implementing minimum first 
        ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
        ulb_ranked = torch.argsort(torch.argsort(unlb_ref))
        ulb_ranked_exp = ulb_ranked.unsqueeze(-1)
        ulb_ranked_diff = torch.abs(ulb_ranked_exp - ulb_ranked_exp.T)
        
        ps_ulb_ranked_sortidx = torch.argsort(ps_ulb_ranked)
        # ulb0ulb0sim_sort = ulb0ulb0sim[:,ps_ulb_ranked_sortidx] 
        # ulb0ulb0sim_sort = ulb0ulb0sim_sort[ps_ulb_ranked_sortidx, :]
        ulb0ulb0sim_sort = ulb0ulb0sim

        # ulb1ulb0sim_sort = ulb1ulb0sim[:,ps_ulb_ranked_sortidx] 
        # ulb1ulb0sim_sort = ulb1ulb0sim_sort[ps_ulb_ranked_sortidx, :]
        ulb1ulb0sim_sort = ulb1ulb0sim

        # ulb0ulb1sim_sort = ulb0ulb1sim[:,ps_ulb_ranked_sortidx] 
        # ulb0ulb1sim_sort = ulb0ulb1sim_sort[ps_ulb_ranked_sortidx, :]
        ulb0ulb1sim_sort = ulb0ulb1sim

        # ulb1ulb1sim_sort = ulb1ulb1sim[:,ps_ulb_ranked_sortidx] 
        # ulb1ulb1sim_sort = ulb1ulb1sim_sort[ps_ulb_ranked_sortidx, :]
        ulb1ulb1sim_sort = ulb1ulb1sim

        #### generate masks
        # mask_diff_generate = torch.arange(ps_ulb_ranked_sortidx.shape[0]).cuda()        
        mask_diff_generate = ps_ulb_ranked
        mask_diff_generate_exp = mask_diff_generate.unsqueeze(-1)
        mask_diff_generate_exp = mask_diff_generate_exp.expand(-1, batch_size)
        mask_diff_generate_diff = mask_diff_generate_exp - mask_diff_generate_exp.T
        mask_diff_upper = (mask_diff_generate_diff < 0).float()
        mask_diff1 = (torch.abs(mask_diff_generate_diff) == 1).float()
        mask_diff1_upr = mask_diff1 * mask_diff_upper
        mask_diff2 = (torch.abs(mask_diff_generate_diff) == 2).float()
        mask_diff2_upr = mask_diff2 * mask_diff_upper
        mask_diff3 = (torch.abs(mask_diff_generate_diff) == 3).float()
        mask_diff3_upr = mask_diff3 * mask_diff_upper
        mask_diff4 = (torch.abs(mask_diff_generate_diff) == 4).float()
        mask_diff4_upr = mask_diff4 * mask_diff_upper
        mask_diff5 = (torch.abs(mask_diff_generate_diff) == 5).float()
        mask_diff5_upr = mask_diff5 * mask_diff_upper
        mask_diff6 = (torch.abs(mask_diff_generate_diff) == 6).float()
        mask_diff6_upr = mask_diff6 * mask_diff_upper
        mask_diff7 = (torch.abs(mask_diff_generate_diff) == 7).float()
        mask_diff7_upr = mask_diff7 * mask_diff_upper
        mask_diff8 = (torch.abs(mask_diff_generate_diff) == 8).float()
        mask_diff8_upr = mask_diff8 * mask_diff_upper
        mask_diff9 = (torch.abs(mask_diff_generate_diff) == 9).float()
        mask_diff9_upr = mask_diff9 * mask_diff_upper
        mask_diff10 = (torch.abs(mask_diff_generate_diff) == 10).float()
        mask_diff10_upr = mask_diff10 * mask_diff_upper
        mask_diff11 = (torch.abs(mask_diff_generate_diff) == 11).float()
        mask_diff11_upr = mask_diff11 * mask_diff_upper
        mask_diff12 = (torch.abs(mask_diff_generate_diff) == 12).float()
        mask_diff12_upr = mask_diff12 * mask_diff_upper
        mask_diff13 = (torch.abs(mask_diff_generate_diff) == 13).float()
        mask_diff13_upr = mask_diff13 * mask_diff_upper
        mask_diff14 = (torch.abs(mask_diff_generate_diff) == 14).float()
        mask_diff14_upr = mask_diff14 * mask_diff_upper
        mask_diff15 = (torch.abs(mask_diff_generate_diff) == 15).float()
        mask_diff15_upr = mask_diff15 * mask_diff_upper
        mask_diff16 = (torch.abs(mask_diff_generate_diff) == 16).float()
        mask_diff16_upr = mask_diff16 * mask_diff_upper
        
        mask_diff_upr_list = [mask_diff1_upr, mask_diff2_upr, mask_diff3_upr, mask_diff4_upr, mask_diff5_upr, mask_diff6_upr, mask_diff7_upr, mask_diff8_upr,mask_diff9_upr, mask_diff10_upr, mask_diff11_upr, mask_diff12_upr, mask_diff13_upr, mask_diff14_upr, mask_diff15_upr, mask_diff16_upr]
        zero_vector = torch.zeros_like(mask_diff1_upr).cuda()
        loss_ulb = torch.tensor(0).float().cuda()
        prcnt_correct_num = torch.tensor(0).float().cuda()
        prcnt_correct_den = torch.tensor(0).float().cuda()
        
        for ulb_loss_itr in range(base_num):
            mask_diff_val_ulb0ulb0_hrz = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb0ulb0sim_sort, dim = -1, keepdim=True).expand(-1, ps_ulb_ranked.shape[0])
            mask_diff_val_ulb0ulb1_hrz = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb0ulb1sim_sort, dim = -1, keepdim=True).expand(-1, ps_ulb_ranked.shape[0])
            mask_diff_val_ulb1ulb0_hrz = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb1ulb0sim_sort, dim = -1, keepdim=True).expand(-1, ps_ulb_ranked.shape[0])
            mask_diff_val_ulb1ulb1_hrz = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb1ulb1sim_sort, dim = -1, keepdim=True).expand(-1, ps_ulb_ranked.shape[0])

            mask_diff_val_ulb0ulb0_vrt = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb0ulb0sim_sort, dim = 0, keepdim=True).expand(ps_ulb_ranked.shape[0], -1)
            mask_diff_val_ulb0ulb1_vrt = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb0ulb1sim_sort, dim = 0, keepdim=True).expand(ps_ulb_ranked.shape[0], -1)
            mask_diff_val_ulb1ulb0_vrt = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb1ulb0sim_sort, dim = 0, keepdim=True).expand(ps_ulb_ranked.shape[0], -1)
            mask_diff_val_ulb1ulb1_vrt = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb1ulb1sim_sort, dim = 0, keepdim=True).expand(ps_ulb_ranked.shape[0], -1)

            ulb_ranked_diff_hrz = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb_ranked_diff, dim = -1, keepdim=True).expand(-1, ps_ulb_ranked.shape[0])
            ulb_ranked_diff_vrt = torch.sum(mask_diff_upr_list[ulb_loss_itr] * ulb_ranked_diff, dim = 0, keepdim=True).expand(ps_ulb_ranked.shape[0], -1)

            # print(mask_diff_val_ulb0ulb0_hrz)
            # print(mask_diff_val_ulb0ulb0_vrt)

            ulb0ulb0sim_sort_hinge_hrz = torch.max(ulb0ulb0sim_sort - mask_diff_val_ulb0ulb0_hrz + hg_margin, zero_vector)
            ulb0ulb1sim_sort_hinge_hrz = torch.max(ulb0ulb1sim_sort - mask_diff_val_ulb0ulb1_hrz + hg_margin, zero_vector)
            ulb1ulb0sim_sort_hinge_hrz = torch.max(ulb1ulb0sim_sort - mask_diff_val_ulb1ulb0_hrz + hg_margin, zero_vector)
            ulb1ulb1sim_sort_hinge_hrz = torch.max(ulb1ulb1sim_sort - mask_diff_val_ulb1ulb1_hrz + hg_margin, zero_vector)

            ulb0ulb0sim_sort_hinge_vrt = torch.max(ulb0ulb0sim_sort - mask_diff_val_ulb0ulb0_vrt + hg_margin, zero_vector)
            ulb0ulb1sim_sort_hinge_vrt = torch.max(ulb0ulb1sim_sort - mask_diff_val_ulb0ulb1_vrt + hg_margin, zero_vector)
            ulb1ulb0sim_sort_hinge_vrt = torch.max(ulb1ulb0sim_sort - mask_diff_val_ulb1ulb0_vrt + hg_margin, zero_vector)
            ulb1ulb1sim_sort_hinge_vrt = torch.max(ulb1ulb1sim_sort - mask_diff_val_ulb1ulb1_vrt + hg_margin, zero_vector)

            ulb_ranked_diff_hrz_hinge = ((ulb_ranked_diff - ulb_ranked_diff_hrz)<0).float()
            ulb_ranked_diff_vrt_hinge = ((ulb_ranked_diff - ulb_ranked_diff_vrt)<0).float()


            mask_diff_edge = (torch.abs(mask_diff_generate_diff) >= ps_ulb_ranked.shape[0] - threshold + ulb_loss_itr).float()
            mask_diff_edge_upr = mask_diff_edge * mask_diff_upper

            ulb0ulb0sim_sort_hinge_msked_hrz = mask_diff_edge_upr * ulb0ulb0sim_sort_hinge_hrz
            ulb0ulb1sim_sort_hinge_msked_hrz = mask_diff_edge_upr * ulb0ulb1sim_sort_hinge_hrz
            ulb1ulb0sim_sort_hinge_msked_hrz = mask_diff_edge_upr * ulb1ulb0sim_sort_hinge_hrz
            ulb1ulb1sim_sort_hinge_msked_hrz = mask_diff_edge_upr * ulb1ulb1sim_sort_hinge_hrz

            ulb0ulb0sim_sort_hinge_msked_vrt = mask_diff_edge_upr * ulb0ulb0sim_sort_hinge_vrt
            ulb0ulb1sim_sort_hinge_msked_vrt = mask_diff_edge_upr * ulb0ulb1sim_sort_hinge_vrt
            ulb1ulb0sim_sort_hinge_msked_vrt = mask_diff_edge_upr * ulb1ulb0sim_sort_hinge_vrt
            ulb1ulb1sim_sort_hinge_msked_vrt = mask_diff_edge_upr * ulb1ulb1sim_sort_hinge_vrt


            
            ulb0ulb0sim_sort_hinge_msked_hrz_vald = (ulb0ulb0sim_sort_hinge_msked_hrz>0).float()
            ulb0ulb1sim_sort_hinge_msked_hrz_vald = (ulb0ulb1sim_sort_hinge_msked_hrz>0).float()
            ulb1ulb0sim_sort_hinge_msked_hrz_vald = (ulb1ulb0sim_sort_hinge_msked_hrz>0).float()
            ulb1ulb1sim_sort_hinge_msked_hrz_vald = (ulb1ulb1sim_sort_hinge_msked_hrz>0).float()

            ulb0ulb0sim_sort_hinge_msked_vrt_vald = (ulb0ulb0sim_sort_hinge_msked_vrt>0).float()
            ulb0ulb1sim_sort_hinge_msked_vrt_vald = (ulb0ulb1sim_sort_hinge_msked_vrt>0).float()
            ulb1ulb0sim_sort_hinge_msked_vrt_vald = (ulb1ulb0sim_sort_hinge_msked_vrt>0).float()
            ulb1ulb1sim_sort_hinge_msked_vrt_vald = (ulb1ulb1sim_sort_hinge_msked_vrt>0).float()

            ulb0ulb0sim_sort_hinge_msked_hrz_vald_crt = ulb0ulb0sim_sort_hinge_msked_hrz_vald * ulb_ranked_diff_hrz_hinge
            ulb0ulb1sim_sort_hinge_msked_hrz_vald_crt = ulb0ulb1sim_sort_hinge_msked_hrz_vald * ulb_ranked_diff_hrz_hinge
            ulb1ulb0sim_sort_hinge_msked_hrz_vald_crt = ulb1ulb0sim_sort_hinge_msked_hrz_vald * ulb_ranked_diff_hrz_hinge
            ulb1ulb1sim_sort_hinge_msked_hrz_vald_crt = ulb1ulb1sim_sort_hinge_msked_hrz_vald * ulb_ranked_diff_hrz_hinge

            ulb0ulb0sim_sort_hinge_msked_vrt_vald_crt = ulb0ulb0sim_sort_hinge_msked_vrt_vald * ulb_ranked_diff_vrt_hinge
            ulb0ulb1sim_sort_hinge_msked_vrt_vald_crt = ulb0ulb1sim_sort_hinge_msked_vrt_vald * ulb_ranked_diff_vrt_hinge
            ulb1ulb0sim_sort_hinge_msked_vrt_vald_crt = ulb1ulb0sim_sort_hinge_msked_vrt_vald * ulb_ranked_diff_vrt_hinge
            ulb1ulb1sim_sort_hinge_msked_vrt_vald_crt = ulb1ulb1sim_sort_hinge_msked_vrt_vald * ulb_ranked_diff_vrt_hinge

            pcorrect_denom = (ulb0ulb0sim_sort_hinge_msked_hrz_vald.sum() + \
                                ulb0ulb1sim_sort_hinge_msked_hrz_vald.sum() + \
                                ulb1ulb0sim_sort_hinge_msked_hrz_vald.sum() + \
                                ulb1ulb1sim_sort_hinge_msked_hrz_vald.sum() + \
                                ulb0ulb0sim_sort_hinge_msked_vrt_vald.sum() + \
                                ulb0ulb1sim_sort_hinge_msked_vrt_vald.sum() + \
                                ulb1ulb0sim_sort_hinge_msked_vrt_vald.sum() + \
                                ulb1ulb1sim_sort_hinge_msked_vrt_vald.sum()+0.00001) 
            pcorrect_num = (ulb0ulb0sim_sort_hinge_msked_hrz_vald_crt.sum() + \
                            ulb0ulb1sim_sort_hinge_msked_hrz_vald_crt.sum() + \
                            ulb1ulb0sim_sort_hinge_msked_hrz_vald_crt.sum() + \
                            ulb1ulb1sim_sort_hinge_msked_hrz_vald_crt.sum() + \
                            ulb0ulb0sim_sort_hinge_msked_vrt_vald_crt.sum() + \
                            ulb0ulb1sim_sort_hinge_msked_vrt_vald_crt.sum() + \
                            ulb1ulb0sim_sort_hinge_msked_vrt_vald_crt.sum() + \
                            ulb1ulb1sim_sort_hinge_msked_vrt_vald_crt.sum()) 

            prcnt_correct_num += pcorrect_num
            prcnt_correct_den += pcorrect_denom


            
            # print(mask_diff_edge_upr.type())
            loss_ulb += (ulb0ulb0sim_sort_hinge_msked_hrz.sum() + \
                            ulb0ulb1sim_sort_hinge_msked_hrz.sum() + \
                                ulb1ulb0sim_sort_hinge_msked_hrz.sum() + \
                                    ulb1ulb1sim_sort_hinge_msked_hrz.sum()) / mask_diff_edge_upr.sum() + \
                        (ulb0ulb0sim_sort_hinge_msked_vrt.sum() + \
                            ulb0ulb1sim_sort_hinge_msked_vrt.sum() + \
                                ulb1ulb0sim_sort_hinge_msked_vrt.sum() + \
                                    ulb1ulb1sim_sort_hinge_msked_vrt.sum()) / mask_diff_edge_upr.sum()

        loss_ulb = loss_ulb / base_num
        

        return loss , ktau, prcnt_correct_num, prcnt_correct_den, loss_ulb




class SupConLoss_admargin_semi_lbulbsplit_ranksim(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_semi_lbulbsplit_ranksim, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels_raw=None, mask=None, dist_raw = None, norm_val = 0.2, scale_s = 150, unlb_ref = None, lambda_val = -1):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        dist_raw_expand = dist_raw.unsqueeze(dim=-1)
        dist_raw_expand = dist_raw_expand.expand(-1, dist_raw_expand.shape[0])

        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_lbub1 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*3:lbulb_bsize*4])/4
        feat_cos_lbulb2 = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_ulbulb = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4])/4


        #### Compute using labeled and ulabled
        # feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        # dist_ulbpbs, labels_ulpbs = get_ulbps(feat_lbulb_cos, labels_raw)
        # labels_ulbpsdornk = labels_ulpbs[labels_ulpbs.shape[0]//2:].view(-1)
        # dist_ulbpsdornk = dist_ulbpbs[dist_ulbpbs.shape[0]//2:].view(-1)        

        # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        # ktau_dist = torch.abs(kendalrankloss(dist_ulbpsdornk, unlb_ref))


        #### Compute using unlabeled only 
        labels_ulpbs = get_ulbps_ulbonly(feat_cos_ulbulb)
        labels_ulbpsdornk = labels_ulpbs    

        ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        ktau_dist = ktau




        labels = labels_raw.contiguous().view(-1, 1)
        dist = dist_raw
        batch_size = labels.shape[0]
        
  
        #### We take labeled only
        feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), 
                                    torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)      

        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        ulb0ulb0sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2]
        ulb0ulb1sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4]
        ulb1ulb0sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2]
        ulb1ulb1sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4]

        
        loss_ulb = torch.tensor(0).float().cuda()

        ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
        batch_unique_targets = torch.unique(ps_ulb_ranked)
        if len(batch_unique_targets) < len(ps_ulb_ranked):
            sampled_indices = []
            for target in batch_unique_targets:
                sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:,0]).item())
            ulb0ulb0sim_samp = ulb0ulb0sim[:,sampled_indices]
            ulb0ulb0sim_samp = ulb0ulb0sim_samp[sampled_indices,:]

            ulb0ulb1sim_samp = ulb0ulb1sim[:,sampled_indices]
            ulb0ulb1sim_samp = ulb0ulb1sim_samp[sampled_indices,:]

            ulb1ulb0sim_samp = ulb1ulb0sim[:,sampled_indices]
            ulb0ulb0sim_samp = ulb1ulb0sim_samp[sampled_indices,:]

            ulb1ulb1sim_samp = ulb1ulb1sim[:,sampled_indices]
            ulb1ulb1sim_samp = ulb1ulb1sim_samp[sampled_indices,:]
            
            ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
        else:
            ulb0ulb0sim_samp = ulb0ulb0sim
            ulb0ulb1sim_samp = ulb0ulb1sim
            ulb1ulb0sim_samp = ulb1ulb0sim
            ulb1ulb1sim_samp = ulb1ulb1sim
            ps_ulb_ranked_samp = ps_ulb_ranked
        
        for i in range(len(ps_ulb_ranked_samp)):
            label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
            # print(ps_ulb_ranked_samp)
            # print(label_ranks)
            # exit()
            feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp[i].unsqueeze(dim=0), lambda_val)
            # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

            feature_ranks_ulb0ulb1 = TrueRanker.apply(ulb0ulb1sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb1, label_ranks)

            feature_ranks_ulb1ulb0 = TrueRanker.apply(ulb1ulb0sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb1ulb0, label_ranks)

            feature_ranks_ulb1ulb1 = TrueRanker.apply(ulb1ulb1sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb1ulb1, label_ranks)
        
        

        return loss , ktau, loss_ulb








class SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, 
                features, 
                labels_raw=None, 
                mask=None, 
                dist_raw = None, 
                norm_val = 0.2, 
                scale_s = 150, 
                unlb_ref = None, 
                lambda_val = -1, 
                ulb_pred_0 = None, 
                ulb_pred_1 = None,
                ulb_pred_avg = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        dist_raw_expand = dist_raw.unsqueeze(dim=-1)
        dist_raw_expand = dist_raw_expand.expand(-1, dist_raw_expand.shape[0])

        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_lbub1 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*3:lbulb_bsize*4])/4
        feat_cos_lbulb2 = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_ulbulb = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4])/4


        #### Compute using labeled and ulabled
        # feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        # dist_ulbpbs, labels_ulpbs = get_ulbps(feat_lbulb_cos, labels_raw)
        # labels_ulbpsdornk = labels_ulpbs[labels_ulpbs.shape[0]//2:].view(-1)
        # dist_ulbpsdornk = dist_ulbpbs[dist_ulbpbs.shape[0]//2:].view(-1)        

        # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        # ktau_dist = torch.abs(kendalrankloss(dist_ulbpsdornk, unlb_ref))


        #### Compute using unlabeled only 
        labels_ulpbs = get_ulbps_ulbonly(feat_cos_ulbulb)
        labels_ulbpsdornk = labels_ulpbs    

        ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        ktau_dist = ktau




        labels = labels_raw.contiguous().view(-1, 1)
        dist = dist_raw
        batch_size = labels.shape[0]
        
  
        #### We take labeled only
        feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), 
                                    torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)      

        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        ulb0ulb0sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2]
        ulb0ulb1sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*3:lbulb_bsize*4]
        ulb1ulb0sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2]
        ulb1ulb1sim = feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4]

        
        loss_ulb = torch.tensor(0).float().cuda()

        ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
        batch_unique_targets = torch.unique(ps_ulb_ranked)
        if len(batch_unique_targets) < len(ps_ulb_ranked):
            sampled_indices = []
            for target in batch_unique_targets:
                sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:,0]).item())
            ulb0ulb0sim_samp = ulb0ulb0sim[:,sampled_indices]
            ulb0ulb0sim_samp = ulb0ulb0sim_samp[sampled_indices,:]

            ulb0ulb1sim_samp = ulb0ulb1sim[:,sampled_indices]
            ulb0ulb1sim_samp = ulb0ulb1sim_samp[sampled_indices,:]

            ulb1ulb0sim_samp = ulb1ulb0sim[:,sampled_indices]
            ulb0ulb0sim_samp = ulb1ulb0sim_samp[sampled_indices,:]

            ulb1ulb1sim_samp = ulb1ulb1sim[:,sampled_indices]
            ulb1ulb1sim_samp = ulb1ulb1sim_samp[sampled_indices,:]
            
            ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
        else:
            ulb0ulb0sim_samp = ulb0ulb0sim
            ulb0ulb1sim_samp = ulb0ulb1sim
            ulb1ulb0sim_samp = ulb1ulb0sim
            ulb1ulb1sim_samp = ulb1ulb1sim
            ps_ulb_ranked_samp = ps_ulb_ranked
        
        for i in range(len(ps_ulb_ranked_samp)):
            label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
            # print(ps_ulb_ranked_samp)
            # print(label_ranks)
            # exit()
            feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp[i].unsqueeze(dim=0), lambda_val)
            # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

            feature_ranks_ulb0ulb1 = TrueRanker.apply(ulb0ulb1sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb1, label_ranks)

            feature_ranks_ulb1ulb0 = TrueRanker.apply(ulb1ulb0sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb1ulb0, label_ranks)

            feature_ranks_ulb1ulb1 = TrueRanker.apply(ulb1ulb1sim_samp[i].unsqueeze(dim=0), lambda_val)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb1ulb1, label_ranks)
        



        loss_ulb_pslb = torch.tensor(0).float().cuda()

        if ulb_pred_avg is not None:
            if len(batch_unique_targets) < len(ps_ulb_ranked):
                assert False, "if len(batch_unique_targets) < len(ps_ulb_ranked) went wrong"
            else:
                # print(ulb_pred_0.shape)
                ulb0ulb0sim_samp_0 =  -torch.abs(ulb_pred_avg - ulb_pred_avg.T)
                ps_ulb_ranked_samp = ps_ulb_ranked
            
            for i in range(len(ps_ulb_ranked_samp)):
                label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
                # print(ps_ulb_ranked_samp)
                # print(label_ranks)
                # exit()
                feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp_0[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

        else:
            if len(batch_unique_targets) < len(ps_ulb_ranked):
                assert False, "if len(batch_unique_targets) < len(ps_ulb_ranked) went wrong"
            else:
                # print(ulb_pred_0.shape)
                ulb0ulb0sim_samp_0 =  -torch.abs(ulb_pred_0 - ulb_pred_0.T)
                ulb0ulb0sim_samp_1 =  -torch.abs(ulb_pred_1 - ulb_pred_1.T)
                ps_ulb_ranked_samp = ps_ulb_ranked
            
            for i in range(len(ps_ulb_ranked_samp)):
                label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
                # print(ps_ulb_ranked_samp)
                # print(label_ranks)
                # exit()
                feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp_0[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

                feature_ranks_ulb0ulb1 = TrueRanker.apply(ulb0ulb0sim_samp_1[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb1, label_ranks)
        

        return loss , ktau, loss_ulb, loss_ulb_pslb







class SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb_sng(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb_sng, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, 
                features, 
                labels_raw=None, 
                mask=None, 
                dist_raw = None, 
                norm_val = 0.2, 
                scale_s = 150, 
                unlb_ref = None, 
                lambda_val = -1, 
                ulb_pred_0 = None, 
                ulb_pred_1 = None,
                ulb_pred_avg = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        dist_raw_expand = dist_raw.unsqueeze(dim=-1)
        dist_raw_expand = dist_raw_expand.expand(-1, dist_raw_expand.shape[0])

        feat_cos_ulbulb = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2]


        #### Compute using labeled and ulabled
        # feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        # dist_ulbpbs, labels_ulpbs = get_ulbps(feat_lbulb_cos, labels_raw)
        # labels_ulbpsdornk = labels_ulpbs[labels_ulpbs.shape[0]//2:].view(-1)
        # dist_ulbpsdornk = dist_ulbpbs[dist_ulbpbs.shape[0]//2:].view(-1)        

        # ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        # ktau_dist = torch.abs(kendalrankloss(dist_ulbpsdornk, unlb_ref))


        #### Compute using unlabeled only 
        labels_ulpbs = get_ulbps_ulbonly(feat_cos_ulbulb)
        labels_ulbpsdornk = labels_ulpbs    

        ktau = torch.abs(kendalrankloss(labels_ulbpsdornk, unlb_ref))
        ktau_dist = ktau




        labels = labels_raw.contiguous().view(-1, 1)
        dist = dist_raw
        batch_size = labels.shape[0]
        
  
        #### We take labeled only
        feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), 
                                    torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], 
                                                feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)      

        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        ulb0ulb0sim = feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2]

        
        loss_ulb = torch.tensor(0).float().cuda()

        ps_ulb_ranked = torch.argsort(torch.argsort(labels_ulbpsdornk))
        batch_unique_targets = torch.unique(ps_ulb_ranked)
        if len(batch_unique_targets) < len(ps_ulb_ranked):
            sampled_indices = []
            for target in batch_unique_targets:
                sampled_indices.append(random.choice((ps_ulb_ranked == target).nonzero()[:,0]).item())
            ulb0ulb0sim_samp = ulb0ulb0sim[:,sampled_indices]
            ulb0ulb0sim_samp = ulb0ulb0sim_samp[sampled_indices,:]

            
            ps_ulb_ranked_samp = ps_ulb_ranked[sampled_indices]
        else:
            ulb0ulb0sim_samp = ulb0ulb0sim
            ps_ulb_ranked_samp = ps_ulb_ranked
        
        for i in range(len(ps_ulb_ranked_samp)):
            label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
            # print(ps_ulb_ranked_samp)
            # print(label_ranks)
            # exit()
            feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp[i].unsqueeze(dim=0), lambda_val)
            # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
            loss_ulb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)        



        loss_ulb_pslb = torch.tensor(0).float().cuda()

        if ulb_pred_avg is not None:
            if len(batch_unique_targets) < len(ps_ulb_ranked):
                assert False, "if len(batch_unique_targets) < len(ps_ulb_ranked) went wrong"
            else:
                # print(ulb_pred_0.shape)
                ulb0ulb0sim_samp_0 =  -torch.abs(ulb_pred_avg - ulb_pred_avg.T)
                ps_ulb_ranked_samp = ps_ulb_ranked
            
            for i in range(len(ps_ulb_ranked_samp)):
                label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
                # print(ps_ulb_ranked_samp)
                # print(label_ranks)
                # exit()
                feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp_0[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

        else:
            if len(batch_unique_targets) < len(ps_ulb_ranked):
                assert False, "if len(batch_unique_targets) < len(ps_ulb_ranked) went wrong"
            else:
                # print(ulb_pred_0.shape)
                ulb0ulb0sim_samp_0 =  -torch.abs(ulb_pred_0 - ulb_pred_0.T)
                ulb0ulb0sim_samp_1 =  -torch.abs(ulb_pred_1 - ulb_pred_1.T)
                ps_ulb_ranked_samp = ps_ulb_ranked
            
            for i in range(len(ps_ulb_ranked_samp)):
                label_ranks = rank_normalised(-torch.abs(ps_ulb_ranked_samp[i] - ps_ulb_ranked_samp).unsqueeze(-1).transpose(0,1))
                # print(ps_ulb_ranked_samp)
                # print(label_ranks)
                # exit()
                feature_ranks_ulb0ulb0 = TrueRanker.apply(ulb0ulb0sim_samp_0[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb0, label_ranks)

                feature_ranks_ulb0ulb1 = TrueRanker.apply(ulb0ulb0sim_samp_1[i].unsqueeze(dim=0), lambda_val)
                # print(feature_ranks_ulb0ulb0.shape, label_ranks.shape)
                loss_ulb_pslb += torch.nn.functional.mse_loss(feature_ranks_ulb0ulb1, label_ranks)
        

        return loss , ktau, loss_ulb, loss_ulb_pslb








class SupConLoss_admargin_val(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_admargin_val, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels_raw=None, mask=None, dist = None, norm_val = 0.2, scale_s = 150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        ### input is (lb,ulb,lb,ulb)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        lbulb_bsize = features.shape[0]

        labels_raw = labels_raw.contiguous().view(-1, 1)


        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # elif self.contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
            # print(anchor_count)
            # exit()

        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T) ### (lb,ulb,lb,ulb) X (lb,ulb,lb,ulb)

        # print(feat_cosim.shape, lbulb_bsize)
        # print(feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2].shape, feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4].shape, feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*4:lbulb_bsize*4].shape)

        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2])/4

        feat_lbulb_cos = feat_cos_lblb0
        kendal_val = get_ulbps_valtau(feat_lbulb_cos, labels_raw)
        labels = labels_raw.contiguous().view(-1, 1)
        batch_size = labels.shape[0]
        # print(batch_size)

        # if not_complex:      
        feat_cosim_ctr = feat_cosim      
        # else:
        #     feat_cosim_ctr = torch.cat((torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1], \
        #                                             feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1]), dim=0), \
        #                                 torch.cat((feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3], \
        #                                             feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3]), dim=0)), dim=1)


        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)
        # print(dist_abdiff)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        ones_fullabdiff = torch.ones_like(dist_fullabdiff)
        
        anchor_dot_contrast = torch.div(feat_cosim_ctr, self.temperature)

        #### potential mismatch? changes shouldn't be too high however 
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        adjn_abdiff = torch.multiply(torch.sub(ones_fullabdiff, mask), dist_fullabdiff)
        adj_abdiff = adjn_abdiff

        anchor_dot_contrast = scale_s* (anchor_dot_contrast + adj_abdiff)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
      
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, kendal_val











class SupConLoss_ctrv2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_ctrv2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, dist = None, norm_val = 0.2, scale_s = 150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            # print(anchor_count)
            # exit()
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        
        # print(dist)   
        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

        feat_sim = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dif_contrast = -2 * feat_sim + 2
        
        mask = mask.repeat(anchor_count, contrast_count)  

        feat_div = torch.multiply(anchor_dif_contrast, dist_fullabdiff)
        feat_div_loss = feat_div.mean()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  
        mask_total = mask.sum()

        feat_sim = mask * anchor_dif_contrast
        feat_sim_loss = feat_sim.sum()/mask_total

        loss = feat_sim_loss - feat_div_loss
        return loss 








class SupConLoss_ctrv2_val(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_ctrv2_val, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, dist = None, norm_val = 0.2, scale_s = 150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            # print(anchor_count)
            # exit()
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        lbulb_bsize = features.shape[0]
        # print(dist)
        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

        feat_sim = torch.matmul(anchor_feature, contrast_feature.T)

        feat_cos_lblb0 = (feat_sim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_sim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_sim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_sim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2])/4

        feat_lbulb_cos = feat_cos_lblb0
        kendal_val = get_ulbps_valtau(feat_lbulb_cos, labels)




        anchor_dif_contrast = -2 * feat_sim + 2
        
        mask = mask.repeat(anchor_count, contrast_count)  

        feat_div = torch.multiply(anchor_dif_contrast, dist_fullabdiff)
        feat_div_loss = feat_div.mean()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask  
        mask_total = mask.sum()

        feat_sim = mask * anchor_dif_contrast
        feat_sim_loss = feat_sim.sum()/mask_total

        loss = feat_sim_loss - feat_div_loss
        return loss , kendal_val











class SupConLoss_ctrv2_semi(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss_ctrv2_semi, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels_raw=None, mask=None, dist = None, norm_val = 0.2, scale_s = 150):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # batch_size = features.shape[0]
        # if labels is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        # elif labels is None and mask is None:
        #     mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # elif labels is not None:
        #     labels = labels.contiguous().view(-1, 1)
        #     if labels.shape[0] != batch_size:
        #         raise ValueError('Num of labels does not match num of features')
        #     mask = torch.eq(labels, labels.T).float().to(device)
        # else:
        #     mask = mask.float().to(device)

        lbulb_bsize = features.shape[0] // 2

        labels_raw = labels_raw.contiguous().view(-1, 1)

        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)


        anchor_feature = contrast_feature
        anchor_count = contrast_count

        
        feat_cosim = torch.matmul(anchor_feature, contrast_feature.T)
        feat_cos_lblb0 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_lbub1 = (feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*2:lbulb_bsize*3, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*0:lbulb_bsize*1, lbulb_bsize*3:lbulb_bsize*4])/4
        feat_cos_lbulb2 = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*0:lbulb_bsize*1] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*2:lbulb_bsize*3] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*2:lbulb_bsize*3])/4
        feat_cos_ulbulb = (feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*1:lbulb_bsize*2] + \
                        feat_cosim[lbulb_bsize*3:lbulb_bsize*4, lbulb_bsize*3:lbulb_bsize*4] + \
                        feat_cosim[lbulb_bsize*1:lbulb_bsize*2, lbulb_bsize*1:lbulb_bsize*2])/4

        feat_lbulb_cos = torch.cat((torch.cat((feat_cos_lblb0, feat_cos_lbub1), dim=0), torch.cat((feat_cos_lbulb2, feat_cos_ulbulb), dim=0)), dim=1)

        dist, labels = get_ulbps(feat_lbulb_cos, labels_raw)
        labels = labels.contiguous().view(-1, 1)
        # print("dist.shape", dist.shape)
        dist = dist.reshape(-1)
        # print("labels.shape", labels.shape)
        batch_size = labels.shape[0]
        # print(batch_size)

        # if not_complex:      
        feat_cosim_ctr = feat_cosim      

        # print(dist)
        dist_expand = dist.unsqueeze(dim=-1)
        dist_expand = dist_expand.expand(-1, batch_size)
        dist_abdiff = torch.clamp(torch.multiply(torch.abs(torch.sub(dist_expand, dist_expand.T)), norm_val),0,2)

        dist_fullabdiff = dist_abdiff.repeat(anchor_count, contrast_count)
        
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

        anchor_dif_contrast = -2 * feat_cosim_ctr + 2
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)  

        feat_div = torch.multiply(anchor_dif_contrast, dist_fullabdiff)
        feat_div_loss = feat_div.mean()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # print(logits_mask.sum())
        mask = mask * logits_mask 
        mask_total = mask.sum()
        # print("mask_total", mask_total)

        feat_sim = mask * anchor_dif_contrast
        feat_sim_loss = feat_sim.sum()/mask_total

        loss = feat_sim_loss - feat_div_loss
        return loss 



