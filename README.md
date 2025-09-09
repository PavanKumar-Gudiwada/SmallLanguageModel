# Small Language Model (SLM) Trained on TinyStories
This repository contains an implementation of a Small Language Model (SLM) trained from scratch on the TinyStories dataset.
The project walks through building and training a transformer-based model end-to-end, including data preprocessing, model design, and inference.

## Project Demo
Follow this link to generate a short story from your own prompt [Demo](https://huggingface.co/spaces/GPavanKumar/ShortStories_SLM).

## Project overview
- Dataset: TinyStories — a dataset of short stories for language model training.
- Goal: Train a transformer-based SLM from scratch for next-token prediction.
- Training: 20,000 epochs with mixed precision training (AMP) and adaptive learning rate scheduling.

## 📂 Workflow
1. Dataset Preparation
- Loaded the TinyStories dataset.
- Preprocessed and tokenized the text data.
- Saved the dataset into train.bin and val.bin for efficient loading.

2. Input Pipeline
- Created input-output pairs using a sliding window approach.
- Prepared sequences suitable for training a transformer model.

3. Model Architecture
Implemented a Transformer-based Small Language Model (SLM) from scratch.
Key components:
- Token embeddings
- Positional encodings
- Multi-head self-attention
- Feed-forward layers
- Layer normalization & residual connections

4. Training Setup
- Loss Function: Negative Log-Likelihood (NLL) / Cross-Entropy.
Optimizer & Scheduler:
- Adaptive learning rate
- Warmup phase followed by cosine decay scheduling.
- Training Enhancements:
- Mixed Precision Training (AMP) for faster computations.
- Gradient scaling for stability.

5. Training
- Trained the model for 20,000 epochs.
- Monitored training and validation loss across epochs.

6. Inference
- Generated text using the trained model.
- Tested with prompts to validate language coherence.

## Requirements
install dependecies with: pip install torch datasets tqdm numpy

## Results
- Successfully trained a transformer-based small language model.
- Demonstrated the ability to learn from scratch on TinyStories.
- Inference results show coherent short text generations. (can be improved by training over more epochs)

## Acknowledgments
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories?utm_source=chatgpt.com)
- Transformer architectures inspired by "Attention Is All You Need".