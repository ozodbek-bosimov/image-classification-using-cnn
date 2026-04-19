# Image Classification using Convolutional Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ozodbek-bosimov/image-classification-using-cnn/blob/main/image_classification_cnn.ipynb)

This project classifies images of **cats and dogs** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Project Structure

```
image-classification-using-cnn/
├── Code/
│   ├── constants.py                # Configuration constants
│   ├── data_prep.py                # Data loading and preprocessing
│   ├── model.py                    # CNN architecture
│   └── main.py                     # Training pipeline (local)
├── image_classification_cnn.ipynb  # Google Colab notebook
├── output/                         # Training results and predictions
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

1. Click the **"Open in Colab"** badge above
2. Go to **Runtime → Change runtime type → GPU (T4)**
3. Run each cell sequentially

## Model Architecture

```
Conv2D(32, 3x3) → MaxPool → BatchNorm
Conv2D(64, 3x3) → MaxPool → BatchNorm
Conv2D(96, 3x3) → MaxPool → BatchNorm
Conv2D(96, 3x3) → MaxPool → BatchNorm → Dropout(0.2)
Conv2D(64, 3x3) → MaxPool → BatchNorm → Dropout(0.2)
Flatten → Dense(256) → Dropout(0.2) → Dense(128) → Dropout(0.3) → Dense(2, softmax)

Loss: categorical_crossentropy | Optimizer: Adam
```

## Results

Trained on **18,000 images** (cats + dogs) for **15 epochs** with 80/20 train/validation split.

| Metric | Score |
|---|---|
| Training Accuracy | 97.59% |
| Validation Accuracy | 90.44% |
| Training Loss | 0.0638 |
| Validation Loss | 0.3255 |

### Training Plots

![Model accuracy (5K images)](./output/accuracy_5000images_15epochs.png)
![Model loss (5K images)](./output/loss_5000images_15epochs.png)
![Model accuracy (18K images)](./output/accuracy_18000images_15epochs.png)
![Model loss (18K images)](./output/loss_18000images_15epochs.png)

### Prediction Samples

![Cat prediction](./output/cat_prediction1.PNG)
![Cat prediction](./output/cat_prediction2.PNG)
![Dog prediction](./output/dog_prediction1.PNG)
![Dog prediction](./output/dog_prediction2.PNG)

## Local Setup

```bash
git clone https://github.com/ozodbek-bosimov/image-classification-using-cnn.git
cd image-classification-using-cnn
pip install -r requirements.txt
```

Download the [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset and extract `train/` and `test1/` folders into the project root, then:

```bash
cd Code/
python main.py
```

## Dataset

[Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) — 25,000 labeled images of cats and dogs.
