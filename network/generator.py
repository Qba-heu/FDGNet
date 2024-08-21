from .discrim_hyperG import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import random
from math import sqrt
import numpy as np



class Mlp(nn.Module):
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

class GlobalFilter_spec(nn.Module):
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0,):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, (dim//2)+1, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.low_or_high = low_or_high

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img_fft, ratio=1.0, noise_mode=1,
                       low_or_high=0, uncertainty_model=0, gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        if low_or_high == 0:
            img_abs = torch.fft.fftshift(img_abs, dim=3)

        c_crop = int(c * sqrt(ratio))
        # h_crop = int(h * sqrt(ratio))
        # w_crop = int(w * sqrt(ratio))
        c_start = c // 2 - c_crop // 2
        # w_start = w - w_crop

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, :, :, c_start:c_start + c_crop], dim=(3), keepdim=True)
                    var = torch.var(img_abs_[:, :, :, c_start:c_start + c_crop], dim=(3), keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, :, :, c_start:c_start + c_crop] = gamma * (
                            img_abs[:, :, :, c_start:c_start + c_crop] - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, :, :, c_start:c_start + c_crop], dim=0, keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, :, :, c_start:c_start + c_crop], dim=0, keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(img_abs[:, :, :, c_start:c_start + c_crop])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(img_abs[:, :, :, c_start:c_start + c_crop]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, :, :, c_start:c_start + c_crop])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img_abs[:, :, :, c_start:c_start + c_crop] = img_abs[:, :, :, c_start:c_start + c_crop] + gamma
        else:
            pass
        if low_or_high == 0:
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))  # recover

        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, layer_index=0, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=3, norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        uncertainty_model=self.uncertainty_model,
                                        gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=C, dim=3, norm='ortho')
        x = x.reshape(B, N, C)
        return x

class GlobalFilter_spa(nn.Module):
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0,):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w-(h//2), dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.low_or_high = low_or_high

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spa_noise(self, img_fft, ratio=1.0, noise_mode=1,
                       low_or_high=0, uncertainty_model=0, gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        if low_or_high == 0:
            img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w - w_crop

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)

                        miu_mean = miu
                        sig_mean = sig

                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = gamma * (
                            img_abs[:, h_start:h_start + h_crop, w_start:, :] - miu) / sig + beta

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

                    if gauss_or_uniform == 0:
                        epsilon_sig = torch.randn_like(img_abs[:, h_start:h_start + h_crop, w_start:, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(img_abs[:, h_start:h_start + h_crop, w_start:, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = img_abs[:, h_start:h_start + h_crop, w_start:,
                                                                        :] + gamma
        else:
            pass
        if low_or_high == 0:
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))  # recover

        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, layer_index=0, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spa_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        uncertainty_model=self.uncertainty_model,
                                        gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class spec_BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5,
                 mask_radio=0.1, mask_alpha=0.5, noise_mode=1, low_or_high=0,
                 uncertainty_model=0, perturb_prob=0.5, uncertainty_factor=1.0,
                 layer_index=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0,):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if layer_index in noise_layers:
            noise_layer_flag = 1
        else:
            noise_layer_flag = 0
        self.filter = GlobalFilter_spec(dim, h=h, w=w,
                                   mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode,
                                   low_or_high=low_or_high, uncertainty_model=uncertainty_model, perturb_prob=perturb_prob,
                                   uncertainty_factor=uncertainty_factor,
                                   noise_layer_flag=noise_layer_flag, gauss_or_uniform=gauss_or_uniform,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.layer_index = layer_index  # where is the block in

    def forward(self, input):
        x = input

        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x), self.layer_index))))
        return x



    def forward(self, input):
        x = input

        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x), self.layer_index))))
        return x

class spa_BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5,
                 mask_radio=0.1, mask_alpha=0.5, noise_mode=1, low_or_high=0,
                 uncertainty_model=0, perturb_prob=0.5, uncertainty_factor=1.0,
                 layer_index=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0,):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if layer_index in noise_layers:
            noise_layer_flag = 1
        else:
            noise_layer_flag = 0
        self.filter = GlobalFilter_spa(dim, h=h, w=w,
                                   mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode,
                                   low_or_high=low_or_high, uncertainty_model=uncertainty_model, perturb_prob=perturb_prob,
                                   uncertainty_factor=uncertainty_factor,
                                   noise_layer_flag=noise_layer_flag, gauss_or_uniform=gauss_or_uniform,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.layer_index = layer_index  # where is the block in

    def forward(self, input):
        x = input

        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x), self.layer_index))))
        return x



class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)

    def forward(self, x,):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


class SpeRandomization(nn.Module):
    def __init__(self,num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap,y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            if y!= None:
                for i in range(len(y.unique())):
                    index= y==y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
        #return (1+gamma)*(x)+beta

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0,low_freq=False):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        self.freq_dis =low_freq
        num_morph = 4
        self.Morphology = MorphNet(imdim)
        self.adain2_morph = AdaIN2d(zdim, num_morph)

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        self.conv1 = nn.Conv2d(n+n+num_morph, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride)
        self.speRandom = SpeRandomization(n)
        self.spaRandom = SpaRandomization(3, device=device)

        self.low_freq_spa = spa_BlockLayerScale(dim=imdim,h=imsize[0],w=imsize[1],uncertainty_model=2)
        self.low_freq_spec = spec_BlockLayerScale(dim=imdim, h=imsize[0], w=imsize[1],uncertainty_model=2)
        self.convFRE = nn.Conv2d(2*imdim, imdim, kernelsize, 1, stride)

    def forward(self, x):
        if self.freq_dis:
            # Encoder of Ours
            B,C,H,W = x.shape
            x1 = x.view(B, C, -1)
            x1 = x1.transpose(1,2)
            x_spa = self.low_freq_spa((x1))
            x_spec = self.low_freq_spec((x1))
            x_spa = x_spa.transpose(1,2)
            x_spa = x_spa.view(B, C, H,W)
            x_spec = x_spec.transpose(1, 2)
            x_spec = x_spec.view(B, C, H, W)
            x = torch.cat((x_spec,x_spa),dim=1)
            x = self.convFRE(x)
        else:
            # Encoder of SDGNet
            x_morph= self.Morphology(x)
            z = torch.randn(len(x), self.zdim).to(self.device)
            x_morph = self.adain2_morph(x_morph, z)

            x_spa = F.relu(self.conv_spa1(x))
            x_spe = F.relu(self.conv_spe1(x))
            x_spa, idx_swap = self.spaRandom(x_spa)
            x_spe = self.speRandom(x_spe,idx_swap)
            x_spe = self.conv_spe2(x_spe)
            x_spa = self.conv_spa2(x_spa)

            x = F.relu(self.conv1(torch.cat((x_spa,x_spe,x_morph),1)))
            x = torch.sigmoid(self.conv2(x))

        return x


