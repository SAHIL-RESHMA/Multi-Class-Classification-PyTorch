### Multi-class Classification with Neural Networks (PyTorch)

This project demonstrates the end-to-end implementation of a multi-class classification model using PyTorch. It involves generating synthetic data, constructing a neural network, training and evaluating the model, and visualizing results. The goal is to classify data points into one of four distinct categories using a deep learning approach.

---

Key Features

- Developed in **Python** using robust libraries such as **PyTorch**, **Scikit-learn**, **Matplotlib**, and **TorchMetrics**
- Simulates a real-world multi-class classification task using generated data
- Builds and trains a neural network from scratch using PyTorch's `nn.Module`
- Employs advanced visualization of decision boundaries
- Includes comprehensive evaluation with metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**

---

Understanding Multi-class Classification

Multi-class classification involves predicting the correct class label for an input where there are more than two possible classes.

In this project:

- **Input Features**: 2 numerical features per data point
- **Output Classes**: 4 unique target classes (labeled 0 to 3)
- **Model Type**: Fully connected feedforward neural network
- **Activation**: Softmax at the output layer to produce probability distribution over classes

---

Dataset Overview

- **Data Source**: Generated using `sklearn.datasets.make_blobs()`
- **Format**: Each data point has 2 numerical features and a class label (0–3)
- **Train/Test Split**: 80/20 using `train_test_split` from Scikit-learn

This synthetic dataset mimics the structure of real-world classification problems, ensuring a diverse class separation for testing decision boundaries.

---

Project Workflow

### 1. Data Preparation
- Generate 1000 data points across 4 centers (classes)
- Visualize data with `matplotlib.pyplot.scatter()`
- Convert to PyTorch `FloatTensor` for compatibility

### 2. Model Construction
- Use `nn.Sequential` to stack layers
- Architecture: `2 → 8 → 8 → 4`
- Activation: ReLU for hidden layers, raw logits (for CrossEntropyLoss) at output

### 3. Training Loop
- Optimizer: **Stochastic Gradient Descent (SGD)** with `lr=0.1`
- Loss Function: **CrossEntropyLoss**
- Epochs: 100 iterations with loss monitoring
- Backpropagation and weight updates handled manually

### 4. Evaluation
- Visual inspection with color-coded prediction plots
- Use of `torchmetrics.Accuracy` and `classification_report` from `sklearn.metrics`
- Display confusion matrix and prediction probabilities

---

### Sample Output
- Test Accuracy: 99.50%
- Sample Prediction:
- Predicted Labels: tensor([1, 3, 2, 1, 0])
- True Labels: tensor([1, 3, 2, 1, 0])

---

### Real-World Applications

- Image recognition (e.g., digit classification)
- Medical diagnostics (e.g., cancer type prediction)
- Customer segmentation in marketing
- Language identification
- Multi-class sentiment analysis
