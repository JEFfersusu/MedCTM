<p align="center">
  <img src="assets/MedCTM_logo.png" width="600px" />
</p>

**🔬 Official PyTorch implementation of "MedCTM: A CNN-Transformer-Mamba Hybrid Network for Medical Image Classification"**


## 🎉 MedCTM Paper Accepted & Published in _Information Processing & Management_!

🔥 **FREE ACCESS LINK** 🔥  
👉 https://authors.elsevier.com/c/1n0TZ_6zz40Md2

---

📄 ScienceDirect:  
https://www.sciencedirect.com/science/article/pii/S030645732600258X

---

[![Overview](https://img.shields.io/badge/Overview-blueviolet?style=for-the-badge&logo=book)]()
[![Architecture](https://img.shields.io/badge/Architecture-orange?style=for-the-badge&logo=building)]()
[![Installation](https://img.shields.io/badge/Installation-green?style=for-the-badge&logo=tools)]()
[![Performance](https://img.shields.io/badge/Performance-red?style=for-the-badge&logo=chart-line)]()
[![Quick Start](https://img.shields.io/badge/QuickStart-brightgreen?style=for-the-badge&logo=chart-line)]()
[![Visualization](https://img.shields.io/badge/Visualization-purple?style=for-the-badge&logo=microscope)]()
[![Applications](https://img.shields.io/badge/Applications-cyan?style=for-the-badge&logo=app)]()

</div>

---

## 📋 Overview

Medical image classification requires models that can **jointly represent local pathological details, long-range spatial dependencies, and their cross-scale interaction**. However, existing approaches struggle to achieve this in a unified and efficient manner. :contentReference[oaicite:0]{index=0}  

**Key Challenges:**
- **CNN-based models** effectively capture local textures but lack explicit modeling of long-range spatial dependencies.
- **Transformer-based models** provide global context modeling but suffer from high computational complexity and limited efficiency in high-resolution medical images.
- **Existing hybrid architectures** rely on loosely coupled fusion (e.g., concatenation or unidirectional attention), failing to model **explicit bidirectional interaction between heterogeneous feature streams**.

**Our Solution:**  
We propose **MedCTM**, a unified hybrid framework that decomposes representation learning into **local modeling, long-range dependency modeling, and cross-branch interaction**, and integrates them through an **interaction-oriented design** rather than static fusion. :contentReference[oaicite:1]{index=1}  

### 🚀 Main Contributions

1. **Bidirectional Channel Interaction Attention (BCIA)**  
   A Transformer-based interaction module that explicitly models **bidirectional conditional dependencies** between CNN (local features) and Mamba (long-range representations), enabling mutual feature refinement rather than static aggregation.

2. **Interaction-Oriented Three-Stage Architecture**  
   A progressive multi-stage design where **CNN, Mamba, and BCIA modules operate in parallel**, enabling iterative feature extraction, interaction, and refinement across scales.

3. **Accuracy–Efficiency Trade-off Optimization**  
   Extensive experiments on **8 datasets spanning 7 imaging modalities** demonstrate that MedCTM achieves **state-of-the-art performance with significantly reduced parameters and computational cost**.

---

## 🏗️ Architecture

<div align="center">

![Bidirectional Channel Interaction Attention](assets/BCIA.png)
*🔄 Bidirectional Channel Interaction Attention Mechanism.*

![MedCTM Framework](assets/framework.png)
*🏗️ Overall Architecture of MedCTM.*

</div>


---

## 🛠️ Installation

### Prerequisites

- Python 3.10 (Ubuntu22.04)
- CUDA 11.8
- PyTorch 2.12

### Step-by-Step Installation

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio
pip install timm==0.9.16 packaging==23.0

pip install triton==2.1.0
pip install causal-conv1d==1.0.1
pip install mamba-ssm==1.0.1

pip install pytest==8.3.5 chardet==4.0.0 yacs==0.1.8 termcolor==2.4.0
pip install scikit-learn==1.3.2 matplotlib==3.7.1
pip install SimpleITK scikit-image PyWavelets==1.4.1
```

---

## 📊 Performance Results

MedCTM achieves state-of-the-art performance across multiple medical imaging benchmarks. Results shown as **Tiny version** / **Large version**.

<div align="center">

| Dataset | Classes| Imaging Modality | F1-Score (%) | AUC (%) | Kappa (%) |
|:--------|:-------:|:------------:|:----------------:|:-------:|:---------:|
| **[Fetal-Planes-DB](https://zenodo.org/records/3904280)** | 4 | Maternal-fetal Ultrasound | **88.8** / **90.1** | **98.8** / **98.9** | **87.8** / **88.7** |
| **[Kvasir v2](https://datasets.simula.no/kvasir/)** | 8 | Gastrointestinal Endoscope | **88.6** / **88.7** | **99.3** / **99.2** | **86.9** / **87.1** |
| **[BloodMNIST](https://medmnist.com/)** | 8 | Blood Cell Microscope | **98.1** / **98.9** | **99.9** / **100.0** | **98.0** / **98.6** |
| **[DermaMNIST](https://medmnist.com/)** | 7 | Dermatoscope | **66.4** / **67.2** | **95.9** / **95.4** | **65.3** / **65.8** |
| **[OrganCMNIST](https://medmnist.com/)** | 11 | Abdominal CT | **89.0** / **89.9** | **99.3** / **99.5** | **87.8** / **89.4** |
| **[OrganSMNIST](https://medmnist.com/)** | 11 | Abdominal CT | **74.9** / **75.9** | **97.7** / **97.9** | **76.5** / **76.9** |
| **[PneumoniaMNIST](https://medmnist.com/)** | 2 | Chest X-Ray | **92.8** / **95.1** | **99.1** / **98.9** | **85.6** / **90.1** |
| **[RetinaMNIST](https://medmnist.com/)** | 5 | Fundus Camera | **42.4** / **43.5** | **74.0** / **75.7** | **37.5** / **37.5** |

</div>



---

## 🚀 Quick Start

### Training Your Model

```bash
# Basic training command
python train.py \
    --model MedCTM_T \
    --dataset PneumoniaMNIST \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### Model Evaluation

```bash
# Evaluate single model
python test.py \
    --dataset PneumoniaMNIST \
    --model MedCTM_T \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 32
```
---

## 🔬 Visualization Results

### Attention Heatmaps

<div align="center">

![Grad-CAM Heatmaps](assets/heatmap.png)
*🔍 Grad-CAM visualization showing model attention on medical images.*

</div>

Our visualizations demonstrate that MedCTM effectively focuses on clinically relevant regions, providing interpretable results for medical professionals.

### Feature Space Analysis

<div align="center">

![t-SNE Results](assets/cluster.png)
*📊 t-SNE visualization comparing feature representations across different models.*

</div>

The t-SNE analysis reveals that MedCTM learns more discriminative feature representations compared to baseline models.

---

## 🔄 Applications & Extensions

MedCTM's flexible architecture enables various medical imaging applications:

### Current Applications
- **Multi-class Medical Image Classification** - Primary focus with demonstrated SOTA results.
- **Binary Classification Tasks** - Excellent performance on disease detection tasks.
- **Cross-domain Medical Imaging** - Robust performance across different imaging modalities.

### Future Extensions
- **🏷️ Generic Image Classification Benchmark** – Designed to assess the model’s ability to generalize across common visual categories.
- **🖼️ Medical Image Segmentation** - Adaptation for pixel-level anatomical structure identification.
- **🎯 Medical Object Detection** - Extension for lesion and abnormality localization.

---


## 🙏 Acknowledgements

We thank but not limited to following repositories for providing assistance for our research:

- **[MedMamba](https://github.com/YubiaoYue/MedMamba)**
- **[MobileFormer](https://github.com/AAboys/MobileFormer)**
- **[MambaOut](https://github.com/yuweihao/MambaOut)**
- **[MobileMamba](https://github.com/lewandofskee/MobileMamba)**
- **[EfficientNetV2](https://github.com/d-li14/efficientnetv2.pytorch)**

Special thanks to the medical imaging community for providing high-quality datasets and benchmarks.

---

<div align="center">

**🔬 MedCTM - Advancing Medical Image Classification through Hybrid Architecture Innovation.**

Made with ❤️ for the medical AI community.

</div>



