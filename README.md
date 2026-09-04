# Practical Guidelines for Pathology VQA

A comparative study of CNN-, attention-, and Transformer-based multimodal architectures for Visual Question Answering (VQA) on the PathVQA dataset, with an additional data-scaling study comparing a CNN and a Transformer visual encoder.

## Overview

This repository contains the implementation and experimental notebooks for our study:

> **Practical Guidelines for Pathology VQA: A Comparative Study of CNNs, RNNs, and Transformers under Data-Scarce Settings**

The study evaluates multimodal architectures that combine pathology image representations with natural-language question representations and investigates how model performance changes with different amounts of training data.

The work is focused on **comparative evaluation and practical guidelines**, rather than proposing a new neural architecture.

## Project Objectives

- Build multimodal VQA systems for pathology image understanding
- Compare CNN-, attention-, and Transformer-based visual architectures
- Study the effect of training-data availability on model performance
- Evaluate open-ended and Yes/No question answering performance
- Analyze multimodal fusion and attention mechanisms for pathology VQA
- Provide practical insights for model selection under data-scarce settings

## Dataset

We use the **PathVQA** dataset introduced by He et al. (2020).

| Property          |  Value |
| ----------------- | -----: |
| Total QA pairs    | 32,799 |
| Total images      |  4,998 |
| Training images   |  2,499 |
| Validation images |  1,499 |
| Test images       |  1,000 |

The experiments use the dataset's official image-level train, validation, and test splits.

### Question Types

- What
- Yes/No
- Where
- How
- When
- Other

## Implemented Methods

### Method 1 — Faster R-CNN + GRU + BAN

- Vision Encoder: Faster R-CNN FPN features
- Text Encoder: GRU
- Fusion: Bilinear Attention Network (BAN)

### Method 2 — EfficientNet-B0 + BiLSTM + Bilinear Fusion

- Vision Encoder: EfficientNet-B0
- Text Encoder: BiLSTM
- Fusion: Bilinear Fusion

### Method 3a — EfficientNet-B3 + LSTM + SAN + LayerNorm

- Vision Encoder: EfficientNet-B3
- Text Encoder: LSTM
- Attention: Stacked Attention Network
- Normalization: LayerNorm

### Method 3b — ResNet50 + LSTM + SAN + LayerNorm

- Vision Encoder: ResNet50
- Text Encoder: LSTM
- Attention: Stacked Attention Network
- Normalization: LayerNorm

## Main Results

| Method    | Architecture                             | Overall EM (%) | Yes/No Accuracy (%) | Open EM (%) | Macro-F1 (%) |
| --------- | ---------------------------------------- | -------------: | ------------------: | ----------: | -----------: |
| Method 1  | Faster R-CNN + GRU + BAN                 |          48.53 |               82.15 |       14.86 |        15.92 |
| Method 2  | EfficientNet-B0 + BiLSTM + Bilinear      |          58.39 |               85.87 |       30.86 |        31.68 |
| Method 3a | EfficientNet-B3 + LSTM + SAN + LayerNorm |          57.85 |           **86.44** |       29.22 |        30.94 |
| Method 3b | ResNet50 + LSTM + SAN + LayerNorm        |      **59.12** |               86.17 |   **32.02** |    **34.49** |

### Additional Evaluation Metrics

| Method    | BLEU-1 (%) | BLEU-2 (%) | BLEU-3 (%) | ROUGE-1 F (%) | ROUGE-L F (%) |
| --------- | ---------: | ---------: | ---------: | ------------: | ------------: |
| Method 1  |      15.55 |       7.11 |       4.70 |         15.93 |         15.92 |
| Method 2  |      31.41 |      14.36 |       9.57 |         31.70 |         31.70 |
| Method 3a |      30.23 |      13.62 |       8.79 |         30.99 |         30.99 |
| Method 3b |  **33.77** |  **16.52** |  **11.19** |     **34.66** |     **34.62** |

## Data-Scaling Study

To study the effect of training-data availability, we compare:

- **ResNet50** as the CNN visual encoder
- **PVTv2-B0** as the Transformer visual encoder

The validation and test sets are kept fixed, while the training set is reduced to different fractions at the image level. All QA pairs associated with selected training images are retained.

| Training Data | ResNet50 Overall EM (%) | PVTv2-B0 Overall EM (%) | ResNet50 Open EM (%) | PVTv2-B0 Open EM (%) |
| ------------- | ----------------------: | ----------------------: | -------------------: | -------------------: |
| 25%           |                   54.29 |                   50.44 |                24.90 |                17.34 |
| 50%           |                   55.81 |                   53.55 |                27.17 |                21.57 |
| 75%           |                   56.99 |                   54.74 |                28.98 |                23.18 |
| 100%          |                   59.12 |                   54.89 |                32.02 |                23.68 |

### CNN vs Transformer Gap

| Training Data | Overall EM Gap | Open EM Gap |
| ------------- | -------------: | ----------: |
| 25%           |          +3.85 |       +7.56 |
| 50%           |          +2.26 |       +5.60 |
| 75%           |          +2.25 |       +5.80 |
| 100%          |          +4.23 |       +8.34 |

The scaling experiment shows a **consistent ResNet50 advantage over PVTv2-B0 across all evaluated training fractions on PathVQA**. The observed advantage is particularly pronounced for open-ended questions.

These findings are specific to the evaluated PathVQA setting and should not be interpreted as a universal claim that CNNs outperform Transformers in all medical VQA or low-data settings.

## Experimental Setup

- Framework: PyTorch
- Hardware: 2 × NVIDIA Tesla T4 (16 GB)
- Random seed: 42
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Training epochs: 15
- Gradient clipping: 1.0
- Vocabulary size: 2,200
- Maximum question length: 30 tokens
- Word embedding dimension: 300
- Image normalization: ImageNet normalization
- Text embeddings: trained from scratch

### Model-Specific Configuration

| Configuration | Batch Size | Image Size | Dropout |
| ------------- | ---------: | ---------: | ------: |
| Method 1      |          4 |  224 × 224 |     0.5 |
| Method 2      |          8 |  224 × 224 |     0.4 |
| Method 3      |          8 |  512 × 512 |     0.4 |

Method 3 additionally uses:

- Label smoothing: 0.1
- Attention hops: 2
- LSTM hidden dimension: 768

## Technologies Used

### Deep Learning

- Python
- PyTorch
- Torchvision
- Hugging Face Transformers

### Models and Architectures

- Faster R-CNN
- EfficientNet-B0
- EfficientNet-B3
- ResNet50
- PVTv2-B0
- GRU
- LSTM
- BiLSTM
- Bilinear Attention Network (BAN)
- Stacked Attention Network (SAN)
- LayerNorm

### Tools

- Kaggle
- Jupyter Notebook
- Git / GitHub

## Repository Structure

```text
Multimodal-PathVQA/
│
├── README.md
├── LICENSE
│
├── methods/
│   ├── Method-01_FasterRCNN_GRU_BAN.ipynb
│   ├── Method-02_EfficientNetB0_BiLSTM_Bilinear.ipynb
│   ├── Method-03a_EfficientNetB3_SAN_LayerNorm.ipynb
│   └── Method-03b_ResNet50_SAN_LayerNorm.ipynb
│
└── scaling/
    ├── ResNet50_25pct.ipynb
    ├── ResNet50_50pct.ipynb
    ├── ResNet50_75pct.ipynb
    ├── PVTv2B0_25pct.ipynb
    ├── PVTv2B0_50pct.ipynb
    ├── PVTv2B0_75pct.ipynb
    └── PVTv2B0_100pct.ipynb
```

## Key Findings

1. **ResNet50 + SAN + LayerNorm** achieves the highest overall Exact Match among the evaluated main models at **59.12%**.
2. **EfficientNet-B3 + SAN + LayerNorm** achieves the highest Yes/No accuracy at **86.44%**.
3. ResNet50 achieves the highest open-ended Exact Match among the evaluated main models at **32.02%**.
4. In the data-scaling experiment, ResNet50 performs better than PVTv2-B0 at every evaluated training fraction.
5. The CNN advantage is more pronounced for open-ended questions than for Yes/No questions.
6. Increasing the training fraction improves ResNet50 performance consistently, while PVTv2-B0 shows much smaller gains between 75% and 100%.
7. The results are specific to the evaluated PathVQA setting and should be interpreted as empirical evidence rather than a universal CNN-versus-Transformer conclusion.

## Conference Presentation

This work was presented as a **poster** at the **2026 International Conference on Sustainable AI for Social Impact and Global Development (SASIGD)**.

The conference is technically co-sponsored by the **IEEE Society on Social Implications of Technology (IEEE SSIT)** and has **IEEE Conference Record #68127**. SASIGD focuses on leveraging AI to address critical global challenges, with emphasis on **sustainability, inclusivity, and equity**.


## References

1. He et al. — _PathVQA: 300K Questions for Medical Visual Question Answering_ (2020)  
   https://arxiv.org/abs/2003.10286
2. Tan and Le — _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_ (2019)
3. He et al. — _Deep Residual Learning for Image Recognition_ (2015)
4. Ren et al. — _Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks_ (2015)
5. Dosovitskiy et al. — _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_ (2020)
6. Wang et al. — _Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions_ (2021)

## Authors

- Madhuri Gottumukkala
- Srivarsha Chivukula
- Peesari Sathvik Reddy

Department of Computer Science and Engineering(Artificial Intelligence & Machine Learning)  
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India

## License

This project is licensed under the MIT License.
