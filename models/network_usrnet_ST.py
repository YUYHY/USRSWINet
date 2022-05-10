import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from utils import utils_image as util
import torch.fft
import models.network_swinir as SW


# for pytorch version >= 1.8.1


"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxH
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        
        # self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.sw_res1 = SW.SwinIR(img_size= 12, patch_size=1, in_chans = nc[3], embed_dim = nc[3],depths=[6,6,6,6] , num_heads=[4, 4, 4, 4], 
                                window_size=3, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.,
                                upsampler="pixelshuffle", resi_connection='1conv')
        #------------------------------
        # swin transform on bridge: image 12, window 3, h/w 4
        # SwinIR -> forward_features_only
        #------------------------------
        self.sw_res2 = SW.SwinIR(img_size= 24, patch_size=1, in_chans = nc[2], embed_dim = nc[2],depths=[6,6,6,6] , num_heads=[4, 4, 4, 4], 
                                window_size=3, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.,
                                upsampler="pixelshuffle", resi_connection='1conv')

        self.sw_res3 = SW.SwinIR(img_size= 48, patch_size=1, in_chans = nc[1], embed_dim = nc[1], depths=[6,6,6,6] , num_heads=[4, 4, 4, 4], 
                                window_size=3, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.,
                                upsampler="pixelshuffle", resi_connection='1conv') 

        self.sw_res4 = SW.SwinIR(img_size= 96, patch_size=1, in_chans = nc[0], embed_dim = nc[0],depths=[6,6,6,6] , num_heads=[4, 4, 4, 4], 
                                window_size=3, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.,
                                upsampler="pixelshuffle", resi_connection='1conv') 

        


        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        
        # self.sw_res2 = 
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        # x.size(): torch.Size([48, 4, 96, 96])

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h) #0
        paddingRight = int(np.ceil(w/8)*8-w) #0
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        # x.size(): torch.Size([48, 4, 96, 96])

        x1 = self.m_head(x) #torch.Size([48, 16, 96, 96]) H
        x2 = self.m_down1(x1) #torch.Size([48, 32, 48, 48]) H/2
        x3 = self.m_down2(x2) #torch.Size([48, 64, 24, 24]) H/4
        x4 = self.m_down3(x3) #torch.Size([48, 64, 12, 12]) H/8
        x = self.sw_res1.forward_features_only(x4)
        # [x,x,41,x] [x,x,42,x]
        # print("--------------debug information---------------------")
        # print("feature size after bridge: ", x.size())
        # print("feature size before bridge: ", x4.size())
        # print("--------------debug information---------------------")
       
        x = self.m_up3(x+x4)


        # print("scale of x after 1st upscale: ", x.size())
        # torch.Size([48, 64, 24, 24])
        x = self.sw_res2.forward_features_only(x)
        x = self.m_up2(x+x3)

        # print("scale of x after 2nd upscale: ", x.size())
        # torch.Size([48, 32, 48, 48])<<--
        x = self.sw_res3.forward_features_only(x) # channel变化
        x = self.m_up1(x+x2)

        # print("scale of x after 3rd upscale: ", x.size())
        # torch.Size([48, 16, 96, 96])
        x = self.sw_res4.forward_features_only(x)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):

        FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + alpha)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
        FX = (FR-FCBinvWBR)/alpha
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2,-1)))

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""


class USRNet_ST(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(USRNet_ST, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        '''
        LR image x: tensor, NxCxWxH            ==torch.Size([48, 3, 32, 32])
        blur kernel k: tensor, Nx(1,3)x w x h  ==torch.Size([48, 1, 25, 25])
        scale factor sf: integer, 1            ==3
        noise level sigma: tensor, Nx1x1x1     ==torch.Size([48, 1, 1, 1])
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))#LR size * scale = HR size
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=sf) # torch.Size([48, 3, 96, 96])
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2,-1)) # torch.Size([48, 3, 96, 96])

        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        #torch.Size([48, 3, 96, 96])


        # hyper-parameter, alpha & beta
        # sigma  Nx1x1x1: torch.Size([48, 1, 1, 1])
        # sf: int 3
        # torch.tensor(sf).type_as(sigma).expand_as(sigma) torch.Size([48, 1, 1, 1])
        # torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1) torch.Size([48, 2, 1, 1])
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))
        # torch.Size([48, 12, 1, 1])
        # interleaved

        # unfolding
        for i in range(self.n):
            
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf) # torch.Size([48, 3, 96, 96])
            # print("input of prior net", torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1).size())           
            # print(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1).size())
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))
        return x
        # ab[:, i+self.n:i+self.n+1, ...] [48,1,1,1]
        # ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3)) torch.Size([48, 1, 96, 96])
        # torch.cat torch.Size([48, 4, 96, 96]) 三个通道是datanet输出，一个通道是hypernet输出 大小和HR相同
