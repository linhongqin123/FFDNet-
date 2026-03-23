import torch
import numpy as np
import sys
import os
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

sys.path.append('.')

from models.network_ffdnet import FFDNet

def main():
    # 强制使用 CPU（避免 GPU 内存问题）
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. 实例化模型
    model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')

    # 2. 加载权重
    pth_path = 'experiments/pretrained_models/ffdnet_gray.pth'
    if not os.path.exists(pth_path):
        print(f"错误：未找到权重文件 {pth_path}")
        return

    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("FFDNet 模型加载成功！")

    # 3. 准备测试图像
    img_path = 'testsets/set12/08.png'   # 请修改为实际路径
    if not os.path.exists(img_path):
        print(f"测试图像不存在：{img_path}")
        return

    # 读取图像并归一化
    img = io.imread(img_path, as_gray=True) / 255.0
    print(f"原始图像尺寸：{img.shape}")

    # 如果图像太大，缩放到最长边 512
    max_size = 512
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = transform.resize(img, (new_h, new_w), anti_aliasing=True)
        print(f"图像已缩放至：{img.shape}")

    sigma_val = 25  # 噪声水平（0-255）
    noisy = img + np.random.normal(0, sigma_val/255.0, img.shape)
    noisy = np.clip(noisy, 0, 1)

    # 转换为 Tensor
    img_t = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)  # shape: (1,1,H,W)

    # 关键修正：sigma_map 形状应为 (1,1,1,1)，表示全局噪声水平
    sigma_map = torch.full((1, 1, 1, 1), sigma_val/255.0, dtype=torch.float).to(device)

    print(f"输入图像形状: {img_t.shape}")
    print(f"噪声图形状: {sigma_map.shape}")

    # 4. 去噪
    with torch.no_grad():
        denoised_t = model(img_t, sigma_map)
    denoised = denoised_t.squeeze().cpu().numpy()
    denoised = np.clip(denoised, 0, 1)

    # 5. 计算 PSNR
    psnr = compare_psnr(img, denoised, data_range=1)
    print(f'去噪后 PSNR = {psnr:.2f} dB')

    # 6. 保存结果
    io.imsave('denoised_output.png', (denoised * 255).astype(np.uint8))
    print("去噪图像已保存为 denoised_output.png")

if __name__ == '__main__':
    main()