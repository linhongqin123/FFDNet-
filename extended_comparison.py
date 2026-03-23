import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import data, img_as_float, io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_wavelet as sk_denoise_wavelet
import bm3d
import os
import sys
import time
sys.path.append('.')
from models.network_ffdnet import FFDNet

# -------------------- 设备 --------------------
device = torch.device('cpu')
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

# -------------------- 去噪函数（带计时）--------------------
def denoise_ffdnet(img, sigma_val, model):
    if model is None: return img, 0
    t0 = time.time()
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    sigma_map = torch.full((1, 1, 1, 1), sigma_val/255.0, dtype=torch.float).to(device)
    with torch.no_grad():
        out = model(img_t, sigma_map)
    denoised = out.squeeze().cpu().numpy()
    elapsed = time.time() - t0
    return denoised, elapsed

def denoise_dncnn(img, model):
    if model is None: return img, 0
    t0 = time.time()
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        residual = model(img_t)
        out = img_t - residual
    denoised = out.squeeze().cpu().numpy()
    elapsed = time.time() - t0
    return denoised, elapsed

def denoise_bm3d(img, sigma_val):
    t0 = time.time()
    img_255 = img * 255.0
    denoised_255 = bm3d.bm3d(img_255, sigma_psd=sigma_val)
    denoised = denoised_255 / 255.0
    elapsed = time.time() - t0
    return denoised, elapsed

def denoise_tv(img, sigma_val):
    t0 = time.time()
    weight = sigma_val / 255.0 * 0.1
    denoised = denoise_tv_chambolle(img, weight=weight, channel_axis=None)
    elapsed = time.time() - t0
    return denoised, elapsed

def denoise_wavelet(img, sigma_val):
    t0 = time.time()
    denoised = sk_denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True)
    elapsed = time.time() - t0
    return denoised, elapsed

# -------------------- 准备测试图像 --------------------
# 使用多张内置图像（可自行添加更多）
image_dict = {
    'camera': data.camera(),
    'coins': data.coins(),
    'checkerboard': data.checkerboard()
}
# 也可以从文件夹读取外部图像，这里为简单使用内置图像

sigma_list = [15, 25, 50]
methods = ['FFDNet', 'DnCNN', 'BM3D', 'TV(ADMM)', 'Wavelet(ISTA/FISTA)']

# 存储结果：results[image_name][sigma][method] = (psnr, ssim, time)
results = {}
for img_name, img in image_dict.items():
    img = img_as_float(img)
    h, w = img.shape
    print(f"\n处理图像: {img_name} ({h}x{w})")
    results[img_name] = {}
    for sigma in sigma_list:
        print(f"  噪声水平: {sigma}")
        # 加噪（固定随机种子以便复现）
        np.random.seed(0)
        noisy = img + (sigma/255.0) * np.random.randn(*img.shape)
        noisy = np.clip(noisy, 0, 1)

        # 去噪并计时
        # FFDNet
        if model_ffdnet:
            den_ffd, t_ffd = denoise_ffdnet(noisy, sigma, model_ffdnet)
            psnr_ffd = psnr(img, den_ffd, data_range=1)
            ssim_ffd = ssim(img, den_ffd, data_range=1)
        else:
            den_ffd, t_ffd = None, 0
            psnr_ffd = ssim_ffd = None

        # DnCNN
        if model_dncnn:
            den_dnc, t_dnc = denoise_dncnn(noisy, model_dncnn)
            psnr_dnc = psnr(img, den_dnc, data_range=1)
            ssim_dnc = ssim(img, den_dnc, data_range=1)
        else:
            den_dnc, t_dnc = None, 0
            psnr_dnc = ssim_dnc = None

        # BM3D
        den_bm3d, t_bm3d = denoise_bm3d(noisy, sigma)
        psnr_bm3d = psnr(img, den_bm3d, data_range=1)
        ssim_bm3d = ssim(img, den_bm3d, data_range=1)

        # TV
        den_tv, t_tv = denoise_tv(noisy, sigma)
        psnr_tv = psnr(img, den_tv, data_range=1)
        ssim_tv = ssim(img, den_tv, data_range=1)

        # Wavelet
        den_wav, t_wav = denoise_wavelet(noisy, sigma)
        psnr_wav = psnr(img, den_wav, data_range=1)
        ssim_wav = ssim(img, den_wav, data_range=1)

        # 存储
        results[img_name][sigma] = {
            'FFDNet': (psnr_ffd, ssim_ffd, t_ffd),
            'DnCNN': (psnr_dnc, ssim_dnc, t_dnc),
            'BM3D': (psnr_bm3d, ssim_bm3d, t_bm3d),
            'TV(ADMM)': (psnr_tv, ssim_tv, t_tv),
            'Wavelet(ISTA/FISTA)': (psnr_wav, ssim_wav, t_wav)
        }

        # 打印当前结果
        print(f"    FFDNet: PSNR={psnr_ffd:.2f}, SSIM={ssim_ffd:.3f}, Time={t_ffd:.2f}s")
        if model_dncnn:
            print(f"    DnCNN:  PSNR={psnr_dnc:.2f}, SSIM={ssim_dnc:.3f}, Time={t_dnc:.2f}s")
        print(f"    BM3D:   PSNR={psnr_bm3d:.2f}, SSIM={ssim_bm3d:.3f}, Time={t_bm3d:.2f}s")
        print(f"    TV:     PSNR={psnr_tv:.2f}, SSIM={ssim_tv:.3f}, Time={t_tv:.2f}s")
        print(f"    Wavelet:PSNR={psnr_wav:.2f}, SSIM={ssim_wav:.3f}, Time={t_wav:.2f}s")

# -------------------- 生成 LaTeX 表格 --------------------
print("\n\n========== LaTeX 表格 ==========")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{不同方法在多个图像和噪声水平下的去噪性能（PSNR/SSIM/时间）}")
print("\\begin{tabular}{c|c|c|cccccc}")
print("\\hline")
print("图像 & $\\sigma$ & 指标 & FFDNet & DnCNN & BM3D & TV-ADMM & Wavelet-ISTA \\\\")
print("\\hline")

for img_name in results:
    for sigma in sigma_list:
        # PSNR 行
        print(f"{img_name} & {sigma} & PSNR & ", end="")
        for method in methods:
            val = results[img_name][sigma].get(method, (None, None, None))[0]
            print(f"{val:.2f} & " if val is not None else "- & ", end="")
        print("\\\\")
        # SSIM 行
        print(f"{img_name} & {sigma} & SSIM & ", end="")
        for method in methods:
            val = results[img_name][sigma].get(method, (None, None, None))[1]
            print(f"{val:.3f} & " if val is not None else "- & ", end="")
        print("\\\\")
        # Time 行
        print(f"{img_name} & {sigma} & Time (s) & ", end="")
        for method in methods:
            val = results[img_name][sigma].get(method, (None, None, None))[2]
            print(f"{val:.2f} & " if val is not None else "- & ", end="")
        print("\\\\")
        print("\\hline")

print("\\end{tabular}")
print("\\label{tab:extended_results}")
print("\\end{table}")

# -------------------- 可选：保存综合对比图 --------------------
# 这里可以生成每个图像的对比图，但为简洁，略