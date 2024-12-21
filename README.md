# Iris-Flower-Classifier-with-PyTorch

## Overview

This project implements a deep learning-based classifier for predicting Iris flower species (Setosa, Versicolor, Virginica) using the PyTorch framework. It demonstrates a simple neural network model to classify flowers based on their sepal and petal dimensions.

## Features

- **Custom Neural Network**: Built from scratch using PyTorch's `nn.Module`.
- **Dataset Handling**: Loads and preprocesses the Iris dataset.
- **Training and Evaluation**: Performs training using a multi-layer perceptron model.
- **Visualization**: Includes visualizations of the dataset and training process.

## Dataset

The Iris dataset is a classic dataset in machine learning containing 150 samples with 4 features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is one of three classes:

1. Setosa (0)
2. Versicolor (1)
3. Virginica (2)

## Requirements

To run this project, you need the following Python libraries:

- PyTorch
- pandas
- matplotlib

Install them using pip:

```bash
pip install torch pandas matplotlib
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/iris-flower-classifier-pytorch.git
   cd iris-flower-classifier-pytorch
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook iris_prediction.ipynb
   ```
3. Follow the steps in the notebook to train and evaluate the model.

## Model Architecture

The neural network model consists of:

- **Input Layer**: Accepts 4 features.
- **Hidden Layer 1**: 8 neurons with ReLU activation.
- **Hidden Layer 2**: 9 neurons with ReLU activation.
- **Output Layer**: 3 neurons (one for each class).

## Dataset Preprocessing

- The dataset is loaded directly from a public URL.
- The target column (`variety`) is converted to numeric values.
- Features (`X`) and labels (`y`) are extracted and converted to NumPy arrays.

## Acknowledgments

The Iris dataset was first introduced by R.A. Fisher and is a staple dataset in the field of machine learning.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

---

Happy coding!

