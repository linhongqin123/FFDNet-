import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_wavelet as sk_denoise_wavelet  # 重命名
import bm3d
import os
import sys
sys.path.append('.')
from models.network_ffdnet import FFDNet

# -------------------- 设备 --------------------
device = torch.device('cpu')  # 使用 CPU 避免 CUDA 兼容性
print(f"Using device: {device}")

# -------------------- DnCNN 模型 --------------------
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bn=True):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        return self.dncnn(x)

# -------------------- 加载模型 --------------------
# DnCNN
dncnn_weights = 'pretrained/model.pth'
if os.path.exists(dncnn_weights):
    checkpoint = torch.load(dncnn_weights, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        model_dncnn = DnCNN().to(device)
        model_dncnn.load_state_dict(checkpoint)
    else:
        model_dncnn = checkpoint.to(device)
    model_dncnn.eval()
    print("DnCNN loaded.")
else:
    model_dncnn = None
    print("DnCNN weights missing.")

# FFDNet
ffdnet_path = 'experiments/pretrained_models/ffdnet_gray.pth'
if os.path.exists(ffdnet_path):
    model_ffdnet = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    state_dict = torch.load(ffdnet_path, map_location=device)
    model_ffdnet.load_state_dict(state_dict, strict=True)
    model_ffdnet = model_ffdnet.to(device)
    model_ffdnet.eval()
    print("FFDNet loaded.")
else:
    model_ffdnet = None
    print("FFDNet weights missing.")

# -------------------- 去噪函数 --------------------
def denoise_ffdnet(img, sigma_val, model):
    if model is None: return img
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    sigma_map = torch.full((1, 1, 1, 1), sigma_val/255.0, dtype=torch.float).to(device)
    with torch.no_grad():
        out = model(img_t, sigma_map)
    return out.squeeze().cpu().numpy()

def denoise_dncnn(img, model):
    if model is None: return img
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        residual = model(img_t)
        out = img_t - residual
    return out.squeeze().cpu().numpy()

def denoise_bm3d(img, sigma_val):
    # BM3D 需要输入 [0,255] 范围，sigma 也是 [0,255]
    img_255 = img * 255.0
    denoised_255 = bm3d.bm3d(img_255, sigma_psd=sigma_val)
    return denoised_255 / 255.0

def denoise_tv(img, sigma_val):
    weight = sigma_val / 255.0 * 0.1  # 可调
    return denoise_tv_chambolle(img, weight=weight, channel_axis=None)

def denoise_wavelet(img, sigma_val):
    # 使用 BayesShrink 自动估计噪声，不显式传递 sigma
    return sk_denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True)

# -------------------- 测试图像 --------------------
image = img_as_float(data.camera())
np.random.seed(0)
sigma_val = 25
noisy = image + (sigma_val/255.0) * np.random.randn(*image.shape)
noisy = np.clip(noisy, 0, 1)

# -------------------- 去噪 --------------------
results = {'Noisy': noisy}
if model_ffdnet is not None:
    results['FFDNet'] = denoise_ffdnet(noisy, sigma_val, model_ffdnet)
if model_dncnn is not None:
    results['DnCNN'] = denoise_dncnn(noisy, model_dncnn)
results['BM3D'] = denoise_bm3d(noisy, sigma_val)
results['TV(ADMM)'] = denoise_tv(noisy, sigma_val)
results['Wavelet(ISTA/FISTA)'] = denoise_wavelet(noisy, sigma_val)

# -------------------- 指标 --------------------
metrics = {}
for name, img in results.items():
    p = psnr(image, img, data_range=1.0)
    s = ssim(image, img, data_range=1.0)
    metrics[name] = (p, s)

# -------------------- 打印表格 --------------------
print("\nMethod\t\t\tPSNR\tSSIM")
print("----------------------------------------")
for name, (p, s) in metrics.items():
    print(f"{name:<20}\t{p:.2f}\t{s:.3f}")

# -------------------- 可视化 --------------------
plt.figure(figsize=(15, 10))
titles = ['Original', 'Noisy'] + list(results.keys())[1:]
images = [image, noisy] + list(results.values())[1:]
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    if title in metrics:
        plt.title(f'{title}\nPSNR={metrics[title][0]:.2f}, SSIM={metrics[title][1]:.3f}')
    else:
        plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig('comparison_full.png', dpi=300)
plt.show()
print("\n对比图像已保存为 comparison_full.png")