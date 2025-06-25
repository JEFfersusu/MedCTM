[//]: # (# MedCTM)
<p align="center">
  <img src="assets/MedCTM_logo_01.png" width="600px" />
</p>

--- 
Official PyTorch implementation of "[**MedCTM: A CNN-Transformer-Mamba Hybrid Network for Medical Image Classification**]".

> **Abstract:** Medical image classification is crucial for computer-aided diagnosis, while existing methods face challenges in jointly modeling local texture, global dependencies, and long-range contextual relationships. Specifically, CNNs typically lack long-range feature capture capability, while Transformers rely on a large amount of labeled data and are computationally expensive. Furthermore, existing hybrid architectures usually suffer from inefficient feature interaction and computational redundancy. To address these problems, in this study, we propose a novel CNN-Transformer-Mamba ternary collaborative architecture named MedCTM networks. The network synergizes convolutional local feature extraction, Transformer-based bidirectional cross-attention, and Mamba for linear complexity long sequence modeling. Firstly, a bidirectional channel interaction attention mechanism dynamically fuses CNN-captured local details with Mamba-modeled global spatial dependencies. Secondly, a lightweight three-tier cascade architecture  is designed to progressively refine multi-scale features. Comprehensive experiments based on eight medical image benchmark datasets demonstrate that MedCTM achieves remarkable state-of-the-art performance and superior computational efficiency.
<div align="center">
  <img src="assets/BCIA.png" width="600px" />
</div>

<div align="center">
  <img src="assets/framework.png" width="1000px" />
</div>

> The overall architecture of MedCTM.


# Installation
* `pip install torch==2.1.2 torchvision==0.16.2 torchaudio`
* `pip install packaging==23.0`
* `pip install timm==0.9.16`
* `pip install pytest==8.3.5 chardet==4.0.0 yacs==0.1.8 termcolor==2.4.0`
* `pip install triton==2.1.0`
* `pip install causal-conv1d==1.0.1`
* `pip install mamba-ssm==1.0.1`
* `pip install scikit-learn==1.3.2 matplotlib==3.7.1`
* `pip install SimpleITK`
* `pip install scikit-image`
* `pip install PyWavelets==1.4.1`
------
# The classification performance of MedCTM
Since MedCTM is suitable for most medical images, you can try applying it to advanced tasks (such as ***multi-label classification***, ***medical image segmentation***, and ***medical object detection***). In addition, we are testing MedCTM with different parameter sizes.



<div align="center" style="font-size: 20px;">

| Dataset | Task | F1-score | AUC | Kappa |
|:------:|:--------:|:----------:|:----------:|:----------:|
| **[Fetal-Planes-DB](https://zenodo.org/records/3904280)** | Multi-Class (4) | **88.8 / 90.1** | **98.8 / 98.9** | **87.8 / 88.7** |
| **[Kvasir](https://datasets.simula.no/kvasir/)** | Multi-Class (8) | **88.6 / 88.7** | **99.3 / 99.2** | **86.9 / 87.1** |
| **[BloodMNIST](https://medmnist.com/)** | Multi-Class (8) | **98.1 / 98.9** | **99.9 / 100.0** | **98.0 / 98.6** |
| **[DermaMNIST](https://medmnist.com/)** | Multi-Class (7) | **66.4 / 67.2** | **95.9 / 95.4** | **65.3 / 65.8** |
| **[OrganCMNIST](https://medmnist.com/)** | Multi-Class (11) | **89.0 / 89.9** | **99.3 / 99.5** | **87.8 / 89.4** |
| **[OrganSMNIST](https://medmnist.com/)** | Multi-Class (11) | **74.9 / 75.9** | **97.7 / 97.9** | **76.5 / 76.9** |
| **[PneumoniaMNIST](https://medmnist.com/)** | Binary-Class (2) | **92.8 / 95.1** | **99.1 / 98.9** | **85.6 / 90.1** |
| **[RetinaMNIST](https://medmnist.com/)** | Multi-Class (5) | **42.4 / 43.5** | **74.0 / 75.7** | **37.5 / 37.5** |

</div>

> On the left is the Tiny version, and on the right is the Large version.
### 📌The next step will be to make the weights public.

# Results visualization

<div align="center">
  <img src="assets/heatmap.png" width="600px" />
</div>

> Heatmaps for visualization based on Grad-CAM. The visualized layers are all from the last layer before entering the classification head.
<div align="center">
  <img src="assets/cluster.png" width="600px" />
</div>

> t-SNE results for MedCTM and other models.

# Get Started

#### Train:

```
python train.py --model resnet18 --epochs 100 --batch_size 32 --lr 0.0001 --dataset PneumoniaMNIST 
```

#### Test:

```
python test.py --dataset PneumoniaMNIST --batch_size 32 --weight_dir best_models --models convnext_tiny resnet18 MedCTM_T
```


# Acknowledgements
We thank but not limited to following repositories for providing assistance for our research:
- [MedMamba](https://github.com/YubiaoYue/MedMamba)
- [MobileFormer](https://github.com/AAboys/MobileFormer)
- [MambaOut](https://github.com/yuweihao/MambaOut)
- [MobileMamba](https://github.com/lewandofskee/MobileMamba)
- [EfficientNetV2](https://github.com/d-li14/efficientnetv2.pytorch)
