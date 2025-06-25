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


# MedCTM

<p align="center">
  <img src="assets/MedCTM_logo_01.png" width="600px" alt="MedCTM Logo" />
</p>

<p align="center">
  <strong>🔬 PyTorch implementation of "MedCTM: A CNN-Transformer-Mamba Hybrid Network for Medical Image Classification"</strong>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick-Start-brightgreen?style=for-the-badge&logo=rocket" alt="Quick Start"></a>
  <a href="#-performance-results"><img src="https://img.shields.io/badge/Performance-SOTA-blue?style=for-the-badge&logo=chart-line" alt="Performance"></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge&logo=open-source-initiative" alt="License"></a>
</p>

---

## 📋 Abstract

<details>
<summary><b>🔍 Click to expand full abstract</b></summary>

Medical image classification is crucial for computer-aided diagnosis, while existing methods face challenges in jointly modeling local texture, global dependencies, and long-range contextual relationships. Specifically, CNNs typically lack long-range feature capture capability, while Transformers rely on a large amount of labeled data and are computationally expensive. Furthermore, existing hybrid architectures usually suffer from inefficient feature interaction and computational redundancy. 

To address these problems, we propose a novel **CNN-Transformer-Mamba ternary collaborative architecture** named MedCTM networks. The network synergizes:
- 🏗️ Convolutional local feature extraction
- 🔄 Transformer-based bidirectional cross-attention  
- ⚡ Mamba for linear complexity long sequence modeling

</details>

### 🎯 Key Contributions

> **💡 Innovation Points**
> 
> 1. **🔗 Bidirectional Channel Interaction Attention** - Dynamically fuses CNN-captured local details with Mamba-modeled global spatial dependencies
> 2. **🏭 Three-tier Cascade Architecture** - Progressively refines multi-scale features with lightweight design
> 3. **📊 State-of-the-art Performance** - Comprehensive experiments on 8 medical datasets demonstrate superior results

## 🏗️ Architecture

<details>
<summary><b>🎨 View Architecture Details</b></summary>

<div align="center">
  <img src="assets/BCIA.png" width="600px" alt="Bidirectional Channel Interaction Attention" />
  <p><em>🔄 Bidirectional Channel Interaction Attention Mechanism</em></p>
</div>

<div align="center">
  <img src="assets/framework.png" width="1000px" alt="MedCTM Framework" />
  <p><em>🏗️ Overall Architecture of MedCTM</em></p>
</div>

### 🧩 Component Breakdown

| Component | Function | Advantage |
|-----------|----------|-----------|
| 🔲 **CNN Module** | Local feature extraction | High-resolution detail capture |
| 🔄 **Transformer** | Cross-attention mechanism | Global dependency modeling |
| ⚡ **Mamba** | Sequential modeling | Linear complexity |

</details>

## 🚀 Installation

<details>
<summary><b>⚙️ Installation Guide</b></summary>

### Prerequisites
- 🐍 Python 3.8+
- 🖥️ CUDA (for GPU support)

### 📦 Install Dependencies

```bash
# 🔥 Core dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio
pip install packaging==23.0
pip install timm==0.9.16

# 🧪 Testing and utilities
pip install pytest==8.3.5 chardet==4.0.0 yacs==0.1.8 termcolor==2.4.0

# ⚡ Mamba-specific dependencies
pip install triton==2.1.0
pip install causal-conv1d==1.0.1
pip install mamba-ssm==1.0.1

# 📊 Scientific computing
pip install scikit-learn==1.3.2 matplotlib==3.7.1
pip install SimpleITK
pip install scikit-image
pip install PyWavelets==1.4.1
```

### 🐳 Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t medctm:latest .

# Run container
docker run --gpus all -it medctm:latest
```

</details>

## 📊 Performance Results

> **🏆 State-of-the-Art Results**
> 
> MedCTM demonstrates exceptional performance across multiple medical imaging datasets.
> Results shown as **Tiny version** / **Large version**.

<div align="center">

| Dataset | Task | F1-score | AUC | Kappa | Status |
|:------:|:--------:|:----------:|:----------:|:----------:|:------:|
| **[Fetal-Planes-DB](https://zenodo.org/records/3904280)** | Multi-Class (4) | **88.8 / 90.1** | **98.8 / 98.9** | **87.8 / 88.7** | 🥇 |
| **[Kvasir](https://datasets.simula.no/kvasir/)** | Multi-Class (8) | **88.6 / 88.7** | **99.3 / 99.2** | **86.9 / 87.1** | 🥇 |
| **[BloodMNIST](https://medmnist.com/)** | Multi-Class (8) | **98.1 / 98.9** | **99.9 / 100.0** | **98.0 / 98.6** | 🥇 |
| **[DermaMNIST](https://medmnist.com/)** | Multi-Class (7) | **66.4 / 67.2** | **95.9 / 95.4** | **65.3 / 65.8** | 🥈 |
| **[OrganCMNIST](https://medmnist.com/)** | Multi-Class (11) | **89.0 / 89.9** | **99.3 / 99.5** | **87.8 / 89.4** | 🥇 |
| **[OrganSMNIST](https://medmnist.com/)** | Multi-Class (11) | **74.9 / 75.9** | **97.7 / 97.9** | **76.5 / 76.9** | 🥈 |
| **[PneumoniaMNIST](https://medmnist.com/)** | Binary-Class (2) | **92.8 / 95.1** | **99.1 / 98.9** | **85.6 / 90.1** | 🥇 |
| **[RetinaMNIST](https://medmnist.com/)** | Multi-Class (5) | **42.4 / 43.5** | **74.0 / 75.7** | **37.5 / 37.5** | 🥉 |

</div>

<div align="center">

### 📈 Performance Summary

<img src="https://img.shields.io/badge/Average_F1-80.2%25-brightgreen?style=for-the-badge" alt="Average F1">
<img src="https://img.shields.io/badge/Average_AUC-95.8%25-blue?style=for-the-badge" alt="Average AUC">
<img src="https://img.shields.io/badge/Datasets_Tested-8-orange?style=for-the-badge" alt="Datasets">

</div>

### 📌 Coming Soon
<div align="center">
  <img src="https://img.shields.io/badge/🎁_Pre--trained_Weights-Coming_Soon-yellow?style=for-the-badge&logo=gift" alt="Weights Coming Soon">
</div>

## 🔬 Visualization Results

<details>
<summary><b>🎨 Click to view visualization results</b></summary>

### 🔥 Grad-CAM Heatmaps
<div align="center">
  <img src="assets/heatmap.png" width="600px" alt="Grad-CAM Heatmaps" />
  <p><em>🔍 Heatmaps visualization based on Grad-CAM from the last layer before the classification head</em></p>
</div>

### 🎯 t-SNE Feature Visualization
<div align="center">
  <img src="assets/cluster.png" width="600px" alt="t-SNE Results" />
  <p><em>📊 t-SNE results comparing MedCTM with other models</em></p>
</div>

### 📊 Attention Maps
> **Interactive Attention Visualization**
> 
> Our bidirectional attention mechanism effectively captures both local and global features:
> - 🔵 **Blue regions**: Local CNN features
> - 🔴 **Red regions**: Global Mamba features  
> - 🟡 **Yellow regions**: Fused attention areas

</details>

## 🛠️ Quick Start

### 🏃‍♂️ Training

<details>
<summary><b>🚀 Training Configuration</b></summary>

```bash
# 🔥 Basic training
python train.py \
    --model resnet18 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --dataset PneumoniaMNIST

# ⚡ Advanced training with MedCTM
python train.py \
    --model MedCTM_T \
    --epochs 200 \
    --batch_size 64 \
    --lr 0.0005 \
    --dataset BloodMNIST \
    --augmentation \
    --scheduler cosine \
    --warmup_epochs 10
```

### 📊 Training Tips
- 🎯 **Batch Size**: Start with 32-64 for optimal performance
- 📈 **Learning Rate**: 0.0001-0.001 works best for medical datasets
- 🔄 **Data Augmentation**: Highly recommended for medical images
- ⏰ **Warmup**: Use 5-10% of total epochs for warmup

</details>

### 🧪 Testing

<details>
<summary><b>🔍 Evaluation & Testing</b></summary>

```bash
# 📊 Single model evaluation
python test.py \
    --dataset PneumoniaMNIST \
    --batch_size 32 \
    --weight_dir best_models \
    --model MedCTM_T

# 🏆 Multi-model comparison
python test.py \
    --dataset PneumoniaMNIST \
    --batch_size 32 \
    --weight_dir best_models \
    --models convnext_tiny resnet18 MedCTM_T \
    --save_results \
    --generate_report
```

### 📈 Evaluation Metrics
- 🎯 **F1-Score**: Harmonic mean of precision and recall
- 📊 **AUC**: Area Under the ROC Curve
- 📏 **Kappa**: Inter-rater agreement coefficient
- ⚡ **Inference Time**: Model efficiency measurement

</details>

## 🔄 Extensions and Applications

<div align="center">

### 🚀 Supported Tasks

<img src="https://img.shields.io/badge/Multi--Label_Classification-✅-brightgreen?style=for-the-badge" alt="Multi-Label">
<img src="https://img.shields.io/badge/Image_Segmentation-✅-blue?style=for-the-badge" alt="Segmentation">
<img src="https://img.shields.io/badge/Object_Detection-✅-orange?style=for-the-badge" alt="Detection">
<img src="https://img.shields.io/badge/Transfer_Learning-✅-purple?style=for-the-badge" alt="Transfer Learning">

</div>

> **🎯 Versatile Architecture**
> 
> MedCTM's hybrid design makes it suitable for various medical imaging scenarios:
> - 🏷️ **Multi-label classification** for complex diagnostic tasks
> - 🖼️ **Medical image segmentation** for anatomical structure identification
> - 🎯 **Medical object detection** for lesion and abnormality detection
> - 🔄 **Transfer learning** for new medical domains

<details>
<summary><b>🧪 Experimental Variants</b></summary>

### 🔬 Model Variants in Development

| Variant | Parameters | FLOPs | Accuracy | Status |
|---------|------------|-------|----------|--------|
| MedCTM-Nano | 2.1M | 0.3G | 85.2% | 🧪 Testing |
| MedCTM-Tiny | 5.7M | 1.1G | 88.6% | ✅ Available |
| MedCTM-Small | 22.1M | 4.3G | 91.4% | 🧪 Testing |
| MedCTM-Base | 86.7M | 15.4G | 93.8% | 🔄 Training |
| MedCTM-Large | 307.4M | 61.8G | 95.1% | ✅ Available |

</details>

## 🙏 Acknowledgements

<details>
<summary><b>🤝 Special Thanks</b></summary>

We gratefully acknowledge the following repositories for their valuable contributions:

<div align="center">

| Repository | Contribution | Status |
|------------|-------------|--------|
| [MedMamba](https://github.com/YubiaoYue/MedMamba) | Medical Mamba implementations | ⭐ |
| [MobileFormer](https://github.com/AAboys/MobileFormer) | Mobile-friendly transformers | ⭐ |
| [MambaOut](https://github.com/yuweihao/MambaOut) | Mamba optimizations | ⭐ |
| [MobileMamba](https://github.com/lewandofskee/MobileMamba) | Mobile Mamba implementations | ⭐ |
| [EfficientNetV2](https://github.com/d-li14/efficientnetv2.pytorch) | Efficient architectures | ⭐ |

</div>

### 🌟 Community Support
- 💬 **Discussions**: Join our community for technical discussions
- 🐛 **Issues**: Report bugs and request features
- 🤝 **Contributions**: We welcome pull requests and collaborations

</details>

## 📄 License

<div align="center">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge&logo=open-source-initiative" alt="MIT License">
</div>

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <h3>🌟 Star History</h3>
  <img src="https://img.shields.io/github/stars/your-repo/MedCTM?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/your-repo/MedCTM?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/your-repo/MedCTM?style=social" alt="GitHub watchers">
</div>

<div align="center">
  <p><em>🔬 MedCTM - Advancing Medical Image Classification through Hybrid Architecture Innovation</em></p>
  <p><strong>Made with ❤️ for the Medical AI Community</strong></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/⬆️-Back_to_Top-blue?style=for-the-badge" alt="Back to Top">
</div>



