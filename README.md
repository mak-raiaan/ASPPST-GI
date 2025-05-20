# Atrous Spatial Pyramid Pooling with Swin Transformer (ASPPST)

This repository provides the official implementation of the **Atrous Spatial Pyramid Pooling with Swin Transformer (ASPPST)** model, developed for the classification of gastrointestinal (GI) tract abnormalities using endoscopic video frames. The framework integrates spatial pyramid pooling with the Swin Transformer architecture to improve classification performance and enhance interpretability.

## 🔬 Overview

The ASPPST model leverages multi-scale feature aggregation through Atrous Spatial Pyramid Pooling (ASPP) and combines it with the Swin Transformer’s hierarchical attention mechanism. This hybrid approach is designed to efficiently capture both local texture and global contextual information, making it well-suited for complex medical image classification tasks.

<p align="center">
  <img src="https://github.com/mak-raiaan/ASPPST-GI/blob/d6ca16dc801f758ab4b8208966313a60717a3f07/GI_model.png" alt="GI Model Architecture" width="600"/>
</p>

## 📁 Repository Structure

- `asppst.py`: Main model architecture and implementation.  
  🔗 [View Code](https://github.com/mak-raiaan/ASPPST-GI/blob/main/asppst.py)

- `Weight/`: Pre-trained weights for ASPPST.  
  🔗 [Download Weights](https://github.com/mak-raiaan/ASPPST-GI/tree/main/Weight)

## 📊 Dataset

The model is trained and evaluated on the **HyperKvasir** dataset, a comprehensive dataset of endoscopic images and videos from the GI tract.

🔗 [Access HyperKvasir Dataset](https://datasets.simula.no/hyper-kvasir/)


## Citation

If you find this work useful for your research, please cite our paper:
```bibtex
@article{abian2025atrous,
  title={Atrous spatial pyramid pooling with swin transformer model for classification of gastrointestinal tract diseases from videos with enhanced explainability},
  author={Abian, Arefin Ittesafun and Raiaan, Mohaimenul Azam Khan and Jonkman, Mirjam and Islam, Sheikh Mohammed Shariful and Azam, Sami},
  journal={Engineering Applications of Artificial Intelligence},
  volume={150},
  pages={110656},
  year={2025},
  publisher={Elsevier}
}
