# Handwritten Digit Recognition Using CNN

## 1. Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images. The model classifies images into ten categories, representing digits from 0 to 9.

This project was developed as part of the CodeAlpha Machine Learning Internship.

## 2. Objectives

- Build a deep learning model for handwritten digit recognition.
- Preprocess image data for training.
- Train and evaluate a CNN using the MNIST dataset.
- Measure model performance using accuracy, precision, recall, and F1-score.
- Save the trained model and predict digits from new images.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## 4. Dataset

The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

Pixel values are normalized from 0–255 to 0–1 before training.

## 5. Model Architecture

The CNN contains:

1. Input layer for 28 × 28 grayscale images
2. Convolutional layer with 32 filters
3. Max-pooling layer
4. Convolutional layer with 64 filters
5. Max-pooling layer
6. Flatten layer
7. Dense layer with 128 neurons
8. Dropout layer with a rate of 0.3
9. Output layer with 10 neurons and softmax activation

The model uses the Adam optimizer and sparse categorical cross-entropy loss.

## 6. Results

The model achieved the following results in the recorded training run:

| Metric | Result |
|---|---:|
| Test accuracy | 98.96% |
| Test loss | 0.0320 |
| Training epochs | 10 |
| Training images | 60,000 |
| Test images | 10,000 |

Precision, recall, and F1-score were approximately 99% across the ten digit classes.

Results may vary slightly between training runs.

## 7. Example Prediction

The saved model was tested on an external image named `my_digit.png`.

- Predicted digit: 7
- Model confidence: 99.54%

Confidence represents the model's predicted probability for the selected class; it is not a guarantee that the prediction is correct.

## 8. Project Files

- `train_model.py` — loads data, builds and trains the CNN, evaluates performance, and saves the model.
- `predict.py` — predicts a digit from an image.
- `handwritten_digit_model.keras` — saved trained model.
- `my_digit.png` — example handwritten digit image.
- `confusion_matrix.png` — model evaluation visualization.
- `mnist_samples.png` — sample MNIST images, if generated.
- `requirements.txt` — Python dependencies.

## 9. Installation

Install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 10. Run the Project

Train and evaluate the model:

```bash
python3 train_model.py
```

Predict a digit from the example image:

```bash
python3 predict.py
```

Ensure that `my_digit.png` and `handwritten_digit_model.keras` are in the project directory before running the prediction script.

## 11. Conclusion

This project demonstrates how convolutional neural networks can be applied to handwritten digit recognition. The model achieved 98.96% test accuracy in the recorded run and successfully recognized an example handwritten digit. The project provides practical experience in image preprocessing, deep learning, model evaluation, and image-based prediction.
