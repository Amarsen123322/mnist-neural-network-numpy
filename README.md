# mnist-neural-network-numpy
A two-layer neural network built with NumPy to classify MNIST handwritten digits.
# MNIST Digit Classifier Using a 2-Layer Neural Network

This project teaches a computer how to recognize handwritten digits from **0 to 9**.

The neural network is built mainly with **NumPy**, so the important parts—forward propagation, loss calculation, backpropagation, and weight updates—are written by hand instead of using a ready-made machine-learning model.

TensorFlow is only used to download the MNIST dataset.

---

## What is MNIST?

MNIST is a famous dataset of handwritten digits.

It contains:

- **60,000 training images**
- **10,000 test images**
- Each image is **28 × 28 pixels**
- Every image contains one digit from **0 to 9**

Example:

```text
Image: handwritten number 5
Label: 5
```

The **image** is the input, and the **label** is the correct answer.

---

## Project Goal

The goal is to train a neural network that can look at a new handwritten digit and predict which number it represents.

For example:

```text
Input image: handwritten 7
Prediction: 7
```

---

## Tools Used

- **Google Colab** — runs the notebook in a web browser
- **Python** — programming language
- **NumPy** — performs mathematical calculations
- **Matplotlib** — displays images and graphs
- **TensorFlow/Keras** — downloads the MNIST dataset

---

## Neural Network Structure

The network has two trainable layers:

```text
784 input values
       ↓
128 hidden neurons
       ↓
10 output neurons
```

### Input

Each MNIST image is originally:

```text
28 × 28 pixels
```

The image is flattened into one row:

```text
28 × 28 = 784 values
```

### Hidden layer

The hidden layer contains **128 neurons**.

It uses the **ReLU activation function**.

### Output layer

The output layer contains **10 neurons**.

Each output represents one digit:

```text
Output 0 → digit 0
Output 1 → digit 1
Output 2 → digit 2
...
Output 9 → digit 9
```

The output layer uses **Softmax** to turn its values into probabilities.

---

## How to Run the Project in Google Colab

### 1. Open Google Colab

Go to Google Colab and create a new notebook.

### 2. Install the packages

Run:

```python
!pip install numpy tensorflow matplotlib
```

### 3. Import the packages

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
```

### 4. Load the dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Check the shapes:

```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

Expected output:

```text
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)
```

---

## Display an MNIST Image

```python
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Correct label: {y_train[0]}")
plt.axis("off")
plt.show()
```

The first training image is usually the digit **5**.

---

## Prepare the Images

### Flatten the images

A basic neural network expects one row of numbers instead of a 28 × 28 square.

```python
x_train_ready = x_train.reshape(60000, 784).astype(np.float32)
x_test_ready = x_test.reshape(10000, 784).astype(np.float32)
```

The new shapes are:

```text
Training: (60000, 784)
Testing:  (10000, 784)
```

### Normalize the pixels

Pixel values originally go from:

```text
0 to 255
```

We divide them by `255` so they become:

```text
0.0 to 1.0
```

```python
x_train_ready = x_train_ready / 255.0
x_test_ready = x_test_ready / 255.0
```

Normalization helps the neural network learn more smoothly.

---

## Prepare the Labels

The original label is one number:

```text
5
```

The network uses **one-hot encoding**, so the label becomes:

```text
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

Create one-hot labels:

```python
y_train_ready = np.zeros((60000, 10), dtype=np.float32)
y_test_ready = np.zeros((10000, 10), dtype=np.float32)

y_train_ready[np.arange(60000), y_train] = 1.0
y_test_ready[np.arange(10000), y_test] = 1.0
```

---

## Initialize the Weights and Biases

Weights are numbers that the network changes while learning.

Biases are extra adjustable values for each neuron.

```python
np.random.seed(42)

input_size = 784
hidden_size = 128
output_size = 10

W1 = np.random.randn(input_size, hidden_size).astype(np.float32)
W1 = W1 * np.sqrt(2.0 / input_size)

b1 = np.zeros((1, hidden_size), dtype=np.float32)

W2 = np.random.randn(hidden_size, output_size).astype(np.float32)
W2 = W2 * np.sqrt(2.0 / hidden_size)

b2 = np.zeros((1, output_size), dtype=np.float32)
```

Parameter shapes:

```text
W1: (784, 128)
b1: (1, 128)
W2: (128, 10)
b2: (1, 10)
```

---

## Activation Functions

### ReLU

ReLU changes negative numbers into zero.

```python
def relu(x):
    return np.maximum(0, x)
```

Example:

```text
Input:  [-3, -1, 0, 2, 5]
Output: [ 0,  0, 0, 2, 5]
```

### ReLU derivative

The derivative is needed during backpropagation.

```python
def relu_derivative(x):
    return (x > 0).astype(np.float32)
```

### Softmax

Softmax changes output scores into probabilities.

```python
def softmax(x):
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exponentials = np.exp(shifted_x)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )
```

All probabilities for one image add up to approximately `1.0`.

---

## Forward Propagation

Forward propagation means sending the image through the neural network.

```python
def forward_pass(x):
    z1 = x @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    probabilities = softmax(z2)

    return z1, a1, z2, probabilities
```

The calculations are:

```text
z1 = input × W1 + b1
a1 = ReLU(z1)

z2 = a1 × W2 + b2
probabilities = Softmax(z2)
```

The digit with the highest probability becomes the prediction:

```python
prediction = np.argmax(probabilities, axis=1)
```

---

## Loss Function

The loss tells us how wrong the prediction is.

- High loss means the prediction is poor.
- Low loss means the prediction is better.
- A loss near zero is very good.

```python
def cross_entropy_loss(probabilities, correct_labels):
    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    return -np.sum(
        correct_labels * np.log(safe_probabilities)
    ) / correct_labels.shape[0]
```

---

## Backpropagation

Backpropagation calculates how each weight and bias should change.

```python
def backward_pass(x, correct_labels, z1, a1, probabilities):
    batch_size = x.shape[0]

    dz2 = (probabilities - correct_labels) / batch_size

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)

    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2
```

The gradient names mean:

```text
dW1 → correction for W1
db1 → correction for b1
dW2 → correction for W2
db2 → correction for b2
```

---

## Update the Parameters

The update rule is:

```text
new value = old value - learning rate × gradient
```

Example:

```python
learning_rate = 0.1

W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1

W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
```

The learning rate controls the size of each update.

---

## Training

Important training terms:

### Batch

A batch is a small group of images processed together.

```text
Batch size: 128 images
```

### Epoch

One epoch means the network has seen all 60,000 training images once.

```text
5 epochs = the network sees all training images 5 times
```

Training settings:

```python
epochs = 5
batch_size = 128
learning_rate = 0.1
```

During training, the network repeatedly:

```text
1. Reads a batch of images
2. Makes predictions
3. Calculates the loss
4. Performs backpropagation
5. Updates weights and biases
6. Moves to the next batch
```

A good training result looks like this:

```text
Epoch 1/5 | Loss: 0.35 | Accuracy: 89.80%
Epoch 2/5 | Loss: 0.17 | Accuracy: 95.10%
Epoch 3/5 | Loss: 0.12 | Accuracy: 96.50%
Epoch 4/5 | Loss: 0.09 | Accuracy: 97.30%
Epoch 5/5 | Loss: 0.07 | Accuracy: 97.80%
```

The exact numbers may be different.

The important signs are:

```text
Loss goes down
Accuracy goes up
```

---

## Testing the Model

The test set contains images the network did not use for learning.

```python
test_z1, test_a1, test_z2, test_probabilities = forward_pass(
    x_test_ready
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

number_correct = np.sum(test_predictions == y_test)
test_accuracy = number_correct / len(y_test)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
```

A simple network like this may reach approximately:

```text
96% to 98% test accuracy
```

The exact result depends on the random starting values and training settings.

---

## Display a Test Prediction

```python
image_number = 0

plt.imshow(x_test[image_number], cmap="gray")
plt.title(
    f"Prediction: {test_predictions[image_number]} | "
    f"Correct: {y_test[image_number]}"
)
plt.axis("off")
plt.show()
```

Change `image_number` to any value from `0` to `9999`.

---

## Common Problems

### `NameError`

Example:

```text
NameError: name 'W1' is not defined
```

This usually means an earlier cell was not run.

Fix:

```text
Runtime → Run all
```

### Shape error

Example:

```text
ValueError: shapes are not aligned
```

Check that:

```text
x_train_ready shape = (60000, 784)
W1 shape = (784, 128)
W2 shape = (128, 10)
```

### The model gives the wrong prediction before training

That is normal. Before training, the weights are random, so the network is guessing.

### Accuracy does not improve

Check that the weights are updated inside the training loop:

```python
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
```

### Training was accidentally reset

Running the weight-initialization cell again erases what the network learned.

Only reset the weights when you want to restart training.

---

## Important Variables

| Variable | Meaning |
|---|---|
| `x_train` | Original training images |
| `y_train` | Original training labels |
| `x_test` | Original test images |
| `y_test` | Original test labels |
| `x_train_ready` | Flattened and normalized training images |
| `x_test_ready` | Flattened and normalized test images |
| `y_train_ready` | One-hot training labels |
| `y_test_ready` | One-hot test labels |
| `W1`, `b1` | First layer parameters |
| `W2`, `b2` | Second layer parameters |
| `probabilities` | Model probabilities for digits 0–9 |
| `test_predictions` | Predicted test digits |

---

## What I Learned

By completing this project, I learned:

- How image data is stored as numbers
- How to flatten and normalize images
- What weights and biases are
- How ReLU and Softmax work
- How forward propagation produces a prediction
- How cross-entropy measures error
- How backpropagation calculates gradients
- How gradient descent updates parameters
- How to train and test a neural network
- How to measure model accuracy

---

## Possible Improvements

Ideas for improving this project:

- Train for more epochs
- Try a different learning rate
- Change the number of hidden neurons
- Add another hidden layer
- Draw graphs of loss and accuracy
- Create a confusion matrix
- Save and reload the trained weights
- Let a user draw a digit and ask the model to predict it

---

## Project Summary

This project builds a handwritten-digit classifier from basic mathematical operations.

The complete learning process is:

```text
Image
  ↓
Flatten and normalize
  ↓
Forward propagation
  ↓
Prediction
  ↓
Calculate loss
  ↓
Backpropagation
  ↓
Update weights
  ↓
Repeat
```

The project is a good introduction to how neural networks learn internally.

---

## Author

Add your name here:

```text
Your Name
```

---

## License

This project is for learning and educational use.
