# Neural Networks Study Notes

---

## 1. Derivatives & Partial Derivatives

### Why they are important

- Derivatives and partial derivatives are **fundamental** for neural network training.
- Used in:
  - **Gradient Descent**
  - **Backpropagation**
  - **Chain Rule**

### Key Concepts

1. **Derivative**:

   - Measures the **rate of change** of a function w.r.t one variable.
   - Function of input → varies with x.
   - Analogy: slope for linear, derivative for non-linear.

2. **Partial Derivative**:

   - Rate of change of a **multivariable function** w.r.t **one variable**, keeping others constant.
   - Essential for updating **weights and biases** in NN.

3. **Slope vs Derivative**
   | Concept | Linear | Non-linear |
   |---------|--------|------------|
   | Slope | Constant | N/A |
   | Derivative | N/A | Function (varies with input) |

---

### Activation Functions & Derivatives

| Function       | Formula                | Derivative         | Notes                          |
| -------------- | ---------------------- | ------------------ | ------------------------------ |
| **Step**       | f(x) = 0 if x<0 else 1 | Not differentiable | Rarely used                    |
| **Sigmoid**    | σ(x) = 1 / (1 + e^-x)  | σ(x)(1-σ(x))       | Smooth, good for probabilities |
| **Tanh**       | tanh(x)                | 1 - tanh²(x)       | Zero-centered                  |
| **ReLU**       | max(0,x)               | 1 if x>0 else 0    | Fast, simple                   |
| **Leaky ReLU** | x if x>0 else αx       | 1 if x>0 else α    | Solves dying ReLU              |

> Knowing derivatives is **essential** for **backpropagation**.

---

## 2. Essential Matrix Operations in Neural Networks

Neural networks rely heavily on **matrix operations** for efficiency.

### Common Operations:

1. **Matrix Multiplication**

   - Computes **weighted sum of inputs**:  
     `Z = X @ W + b`  
     Where:
     - X: input vector/matrix
     - W: weight matrix
     - b: bias vector

2. **Transpose**

   - Needed to align shapes for matrix multiplication during backpropagation.

3. **Element-wise operations**

   - Apply activation functions to all elements of a matrix:
     - ReLU, Sigmoid, Tanh, etc.

4. **Dot Product**

   - Single output neuron: weighted sum of inputs → dot product of vectors.

5. **Gradient computation**
   - Partial derivatives are computed using **matrix operations** for all weights at once.

---

### Notes:

- Matrix operations make **forward pass** and **backpropagation** fast.
- Shapes must align:
  - X: (batch_size × input_features)
  - W: (input_features × output_features)
  - Z = X @ W → (batch_size × output_features)

---

### Quick Tip:

- Always visualize shapes when doing matrix operations in NN.
- Use **NumPy** for efficient calculations:

```python
Z = np.dot(X, W) + b
A = sigmoid(Z)
dW = np.dot(X.T, dZ) / m
```


# 3. Loss (Cost) Functions in Neural Networks

Loss or cost functions measure **how well a neural network is performing**.  
They quantify the **difference between predicted values and true values**, and are **essential for training** (used in gradient descent and backpropagation).

---

##  Common Loss Functions

### **1. Mean Absolute Error (MAE)**
- Formula:  
  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **Intuition:** Average absolute difference between predictions and true values.
- **Pros:** Robust to outliers.
- **Cons:** Less sensitive to large errors compared to MSE.
- **Use Case:** Regression problems.

---

### **2. Mean Squared Error (MSE)**
- Formula:  
  \[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
- **Intuition:** Average of squared differences. Penalizes larger errors more.
- **Pros:** Differentiable and smooth → good for optimization.
- **Cons:** Sensitive to outliers.
- **Use Case:** Regression problems.

---

### **3. Binary Cross-Entropy (Log Loss)**
- Formula:  
  \[
  BCE = - \frac{1}{n} \sum_{i=1}^{n} \big[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\big]
  \]
- **Intuition:** Measures error in probability predictions for binary classification.
- **Pros:** Works well with probabilistic outputs.
- **Use Case:** Binary classification, e.g., predicting 0 or 1.

---

## Python Implementation

```python
import numpy as np

# Sample data
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.2])

# Mean Absolute Error
mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)

# Mean Squared Error
mse = np.mean((y_true - y_pred)**2)
print("MSE:", mse)

# Binary Cross-Entropy (Log Loss)
epsilon = 1e-15  # to avoid log(0)
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
bce = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
print("Binary Cross-Entropy:", bce)
```