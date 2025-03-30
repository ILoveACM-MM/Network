<div align="center">
<h1> RoDeCon-Net: Robust Medical Image Segmentation via Foreground-Background-Uncertain Region Decoupling and Contrast-Driven Feature Enhancement </h1>
</div>

## 🎈 News

- [2025.2.19] Training and inference code released

## ⭐ Abstract

Medical image segmentation faces numerous challenges, particularly the impact of ambiguous transition regions (soft boundaries) between foreground and background on segmentation results. Additionally, the coexistence of salient targets and non-salient targets in low-contrast and blurred-background scenarios is another issue that needs to be addressed.
To tackle these problems, we propose Net, which contains three key components: Feature Decoupling Unit (FDU), Contrast-driven Feature Alignment Unit (CFAU), and Cross-layer Feature Cascade Unit (CFCU).	
The FDU dynamically decouples encoded features into foreground, background, and uncertain regions through forward attention, backward attention, and difference-aware mechanisms. By integrating foreground loss, background loss, and uncertainty loss functions, the FDU guides the model to better distinguish between foreground and background while reducing the interference caused by uncertain regions, thereby enhancing the detection accuracy of salient target boundaries.	
The CFAU further utilizes the foreground and background features extracted by the FDU for multi-layer feature fusion, enhancing the model's ability to detect salient targets in complex background environments.	
Furthermore, to strengthen the model's multiscale perception capability, we introduce the CFCU. This unit not only achieves efficient aggregation of shallow texture details and deep global semantics but also enhances the multiscale representation ability of each layer, effectively addressing the challenges posed by the coexistence of salient and non-salient targets.	
Experimental results show that Net outperforms 13 existing advanced methods on 5 public datasets, demonstrating its superior performance and broad applicability in medical image segmentation tasks.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="figures/challenge.png?raw=true">
</div>

The challenges: (a) the effects of blurred edges (soft boundaries). (b) the difficulty in distinguishing between salient and non-salient objects.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="figures/network.png?raw=true">
</div>

Illustration of the overall architecture.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n Net python=3.8
conda activate Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), BUSI from this [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018), and COVID-19 from this [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the Net

```
python train.py --datasets ISIC2018
training records is saved to ./log/dataset folder
pre-training file is saved to ./log/dataset folder
concrete information see train.py, please
```

### 3. Test the RoMER-UNet

```
python test.py --datasets ISIC2018
training records is saved to ./log/dataset folder
pre-training file is saved to ./log/dataset folder
concrete information see test.py, please
```

### 3. Code example

```
python Test/test_example.py
```

## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="figures/com_pic.png?raw=true">
</div>

<div align="center">
We compare our method against 13 state-of-the-art methods. The red box indicates the area of incorrect predictions.
</div>

## ✨ Quantitative comparison

<div align="center">
<img width="800" alt="image" src="figures/com_tab.png?raw=true">
</div>

<div align="center">
Performance comparison with ten SOTA methods on ISIC2018, Kvasir, BUSI, COVID-19 and Monu-Seg datasets.
</div>

## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveACM-MM/Network/blob/main/LICENSE).