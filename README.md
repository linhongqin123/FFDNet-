# 基于FFDNet的图像去噪与对比研究

本仓库包含论文《基于FFDNet的图像去噪与对比研究》的LaTeX源码及相关实验数据。该工作评估了快速灵活的CNN去噪方法FFDNet与传统及深度学习方法（DnCNN、BM3D、TV-ADMM、小波阈值）的性能。

##  摘要

本文研究了FFDNet（一种快速灵活的基于CNN的去噪方法）与几种经典及深度学习方法（包括DnCNN、BM3D、TV-ADMM和小波阈值）的性能。在多种图像和不同噪声水平上的实验表明，FFDNet和DnCNN取得了最高的PSNR和SSIM，而BM3D以中等速度提供了有竞争力的性能。本文还详细描述了用于TV去噪的ADMM算法。

##  对比方法

- **FFDNet** – 基于CNN的去噪器，可接受噪声水平图作为输入，单个模型即可处理任意噪声水平。
- **DnCNN** – 用于高斯去噪的深度CNN残差学习。
- **BM3D** – 经典的变换域协同滤波方法。
- **TV-ADMM** – 通过交替方向乘子法求解的全变分去噪。
- **小波阈值** – 基于BayesShrink的小波去噪。

##  实验设置

- 测试图像：camera、coins、checkerboard（灰度图）
- 噪声：加性高斯白噪声，标准差 σ = 15、25、50
- 评价指标：PSNR (dB)、SSIM
- 运行时间：在Intel i7-12700H CPU（无GPU加速）上记录

##  实验结果

所有图像和噪声水平下的平均性能：

| 方法            | PSNR (dB) | SSIM  | 时间 (s) |
|----------------|-----------|-------|----------|
| FFDNet         | 29.7      | 0.799 | 0.09     |
| DnCNN          | 28.7      | 0.733 | 0.33     |
| BM3D           | 29.2      | 0.786 | 1.12     |
| TV-ADMM        | 22.9      | 0.483 | 0.03     |
| 小波阈值        | 26.3      | 0.639 | 0.01     |

详细性能曲线和可视化比较请参见论文原文。

##  环境要求

编译LaTeX文档需要TeX发行版（TeX Live、MikTeX）及以下宏包：

- `geometry`
- `mathptmx`, `microtype`
- `xcolor`
- `amsmath`, `amssymb`, `amsthm`, `bm`
- `titlesec`
- `fancyhdr`
- `graphicx`, `caption`, `float`, `booktabs`, `array`
- `hyperref`
- `enumitem`

##  使用说明

1. 克隆本仓库：
   ```bash
   git clone https://github.com/linhongqin123/FFDNet-.git
   ```
2. 确保所有图片文件（`ffdnet_diagram.pdf`、`psnr_curves.png`等）与`.tex`文件放在同一目录下。

   ```
