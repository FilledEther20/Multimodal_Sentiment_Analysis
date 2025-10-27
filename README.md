# 🧠 Multimodal Sentiment & Emotion Recognition using MELD Dataset

This repository contains an advanced **multimodal deep learning model** that performs **sentiment and emotion classification** from video data by fusing textual, visual, and audio modalities. The model is trained and evaluated on the [**MELD (Multimodal EmotionLines Dataset)**](https://www.kaggle.com/datasets/zaber666/meld-dataset), a benchmark dataset for multimodal emotion recognition in conversations.

---

## 📘 Overview

Human emotion understanding requires analyzing not just words, but also **tone** and **facial expressions**. This project tackles the challenge by combining **textual**, **audio**, and **visual** features into a unified model capable of predicting:

- 🎭 **Emotions** — *anger, disgust, fear, joy, neutral, sadness, surprise*  
- 😊 **Sentiments** — *negative, neutral, positive*

The model leverages **pretrained encoders** for each modality and a **fusion-based neural architecture** to capture cross-modal interactions effectively.

---

## 🧩 Model Architecture

The system consists of three independent modality encoders followed by a **fusion and classification** stage.

### **1. Text Encoder**
- Based on **BERT (bert-base-uncased)** from HuggingFace Transformers.
- Uses the `[CLS]` token representation for contextual text embeddings.
- Parameters are frozen to prevent overfitting on small text datasets.
- Projects 768-dimensional BERT embeddings into a 128-dimensional latent space.

```python
BERT → Linear(768 → 128)
```
---

## 2. Video Encoder

* Uses **ResNet3D (r3d_18)** pretrained on Kinetics-400 from `torchvision.models.video`.
* Learns spatiotemporal visual patterns from sequences of video frames.
* Output projection: `Linear → ReLU → Dropout`, producing a 128-dimensional representation.
```
r3d_18(pretrained=True) → Linear(fc_features → 128)
```

---

## 3. Audio Encoder

* A lightweight 1D CNN that extracts temporal features from pre-extracted audio spectrogram embeddings.
* Two convolutional layers with batch normalization, pooling, and an adaptive average pooling.
* Final projection layer reduces to a 128-dimensional vector.
```
Conv1d → BatchNorm → ReLU → Pool → Conv1d → BatchNorm → ReLU → AvgPool → Linear(128)
```

---

## 4. Multimodal Fusion & Classification

After encoding, the three modalities are concatenated and passed through a fusion network and two classification heads.

### Fusion Layer:
```
Linear(128 * 3 → 256) → BatchNorm → ReLU → Dropout
```

### Classification Heads:

* **Emotion Classifier:** 256 → 64 → 7
* **Sentiment Classifier:** 256 → 64 → 3

Both use ReLU activations and dropout for regularization.

---

## 🧮 Training Details

* **Optimizer:** Adam (modality-specific learning rates)
* **Loss:** Weighted CrossEntropyLoss with label smoothing (0.05)
* **Learning Rate Scheduler:** ReduceLROnPlateau
* **Gradient Clipping:** `max_norm=1.0`
* **Metrics:** Accuracy and Precision (weighted)
* **Logging:** TensorBoard (`runs/` directory)

### Class Weight Computation

Class weights for both emotion and sentiment categories are calculated dynamically from the training dataset to handle class imbalance.

---

## 🧠 Dataset — MELD

MELD is a multimodal emotion recognition dataset containing dialogues from the TV show Friends. Each utterance includes:

* 📝 **Text** (transcript)
* 🎥 **Video** (face-focused clip)
* 🎧 **Audio** (voice recording)
* 🎯 **Emotion label** (7 classes)
* 😄 **Sentiment label** (3 classes)

### Example fields in the CSV file:
```
Utterance_ID, Dialogue_ID, Emotion, Sentiment, Speaker, Text
```

### Expected directory structure:
```
dataset/
├── train/
│   ├── train_sent_emo.csv
│   └── train_splits/
├── dev/
├── test/
```

---

## 🚀 How It Works

1. Each utterance is processed into:
   * **Text inputs** → Tokenized via BERT tokenizer
   * **Video frames** → Loaded and resized into tensors
   * **Audio features** → Precomputed spectrograms or embeddings

2. The three encoders transform their respective modalities into 128-dimensional vectors.

3. The fused representation (384-dim) is passed through the fusion network and then to two classifiers:
   * One predicts **emotion**
   * The other predicts **sentiment**

4. During training, both outputs contribute to the total loss.

---

## 🧪 Example Inference
```python
model = MultimodalSentimentModel()
model.eval()

with torch.inference_mode():
    outputs = model(text_inputs, video_frames, audio_features)

emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]
```

### Output Example:
```
anger: 0.12
joy: 0.63
neutral: 0.18
...
positive: 0.70
neutral: 0.20
negative: 0.10
```

---

## Summary

This multimodal architecture leverages:
- **BERT** for contextual text understanding
- **ResNet3D** for spatiotemporal video analysis
- **1D CNN** for audio feature extraction
- **Fusion network** for cross-modal integration
- **Dual classification heads** for emotion and sentiment prediction

The model is trained end-to-end with class balancing, label smoothing, and adaptive learning rate scheduling to achieve robust performance on the MELD dataset.
