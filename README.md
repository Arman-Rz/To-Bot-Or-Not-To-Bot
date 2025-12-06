# To Bot or Not to Bot — NLP/MM Project

**Author:** Arman Rashidizadeh  
**Date:** December 2025  
**Project:** Natural Language Processing

## Project Overview

This project investigates the detection of **AI-generated** or **automated fake personas** on social media by analyzing the **the coherence and authenticity** of their multimodal footpront:

- Linguistic patterns (posts, bios)
- Visual consistency (profile images)
- Behavioral cues (presence or absence of images)

The goal is to build a **multimodal classifier** that can predict whether a social media account is **human** or **bot** using both textual and visual signals.

This project was developed as part of a university coursework in **Natural Language Processing**.

## Repository Structure

```
.
├── data/
|   └── twibot22/
|
├── notebooks/
│   ├── 01_data_cleaning_twibot22.ipynb
│   ├── 02_roberta_finetuning.ipynb
│   ├── 03_clip_multimodal.ipynb
│   └── 04_multimodal_fusion.ipynb
│
|── requirements.txt
└── README.md

```

## Datasets

**TwiBot-22**
This project uses the **TwiBot-22** dataset. Due to licensing restrictions, the dataset cannot be redistributed here.  
Please download it from the official repository:
[TwiBot-22 Dataset](https://github.com/LuoUndergradXJTU/TwiBot-22)  
If you use TwiBot-22 in your work, please cite the authors:

```
@inproceedings{fengtwibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and others},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

## Notebooks

### Data Cleaning & Preprocessing

**`01_data_cleaning_twibot22.ipynb`**  
In this notebook, we load the users, labels and tweets (only one of the tweets files) and extract user metadata, aggregated tweet text and binary labels. In the end, two clean CSVs are produced:

- `users.csv`
- `posts.csv`

### Text Modeling with RoBERTa

**`02_roberta_finetuning.ipynb`**

- Builds a user-level document: `bio + tweets`
- Tokenizes and fine-tunes **RoBERTa-base** on GPU
- Outputs:
  - Accuracy
  - Macro F1
  - AUROC
- Saves the fine-tuned model
  **Best AUROC achieved: ~0.72**

### Image Modeling with CLIP

**`03_clip_multimodal.ipynb`**

- Loads user profile images
- Compute Cosine similarity between image and text embeddings
- Extracts 512-dim visual embeddings using **CLIP ViT-B/32**
- Trains a light classifier on top of CLIP features
- Saves embeddings

### Multimodal Fusion (Text + Image)

**`04_multimodal_fusion.ipynb`**  
combines:

- RoBERTa text probability
- CLIP image probability
- `has_image` feature
- Train using **XGBoost**

## Installation

1.  Clone the repo:
    ```bash
        git clone https://github.com/Arman-Rz/To-Bot-or-Not-To-Bot.git
        cd To-Bot-or-Not-To-Bot
    ```
2.  Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```
3.  Download datasets manually
    Due to licensing, download dataset from the provided link, nad place them under:
   ```  
    ├── data/
    |  └── twibot22/
    |  ├── users.json
    |  ├── labels.csv
    |  └── tweet_0.json
   ```

4.  Run the Project
    To fully reproduce all the results reported in this study, run the notebooks in the following order: 1. `01_data_cleaning_twibot22.ipynb` 2. `02_roberta_finetuning.ipynb` 3. `03_clip_multimodal.ipynb` 4. `04_multimodal_fusion.ipynb`

## Results Summary

| Model          | Modality     | Accuracy | Macro F1 | AUROC |
| -------------- | ------------ | -------- | -------- | ----- |
| CLIP           | Image-only   | 0.474    | 0.474    | 0.465 |
| RoBERTa        | Text-only    | 0.646    | 0.643    | 0.754 |
| XGBoost Fusion | Text + Image | 0.646    | 0.644    | 0.739 |

### Reproducibility Note

This project involves fine-tuning RoBERTa and training multimodal fusion models.  
Due to the use of GPU operations and several sources of randomness (tokenization order, PyTorch CUDA kernels, data shuffling, and XGBoost behavior), **results may vary slightly between runs**, even when using fixed random seeds.  
Clearing the HuggingFace cache or re-running the full pipeline from scratch can also lead to small differences, because the data filtering order may change before splitting into train/val/test sets.  
These variations are normal and typically affect metrics by a small amount.
