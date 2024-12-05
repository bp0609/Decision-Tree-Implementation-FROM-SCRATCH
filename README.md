# DecisionTree-Library: A Custom Implementation of Decision Tree Algorithm

## Overview

**DecisionTree-Library** is a Python-based implementation of a Decision Tree model designed for educational and experimental purposes. The library includes modules for training and evaluating decision trees on various datasets, along with utilities for data preprocessing and visualization. The repository also demonstrates the application of the decision tree model to real-world problems like automotive efficiency analysis, comparing its performance with Scikit-learn’s implementation.

---

## Features

### 1. Decision Tree Class (`tree/base.py`)
- **Initialization:** Allows setting hyperparameters like maximum tree depth and minimum samples per split.
- **Training (`fit` method):** Builds the decision tree using recursive splitting.
- **Prediction (`predict` method):** Predicts target values for unseen data.
- **Splitting:** Implements data splitting based on the chosen performance metric.
- **Stopping Criteria:** Includes mechanisms to prevent overfitting by limiting tree depth or minimum split size.

### 2. Performance Metrics (`tree/metrics.py`)
- Implemented metrics for evaluating splits:
  - **Gini Index** (for classification)
  - **Entropy** (for classification)
  - **Mean Squared Error (MSE)** (for regression)
- Helper functions to calculate these metrics efficiently.

### 3. Utilities (`tree/utils.py`)
- Functions for:
  - **Data Preprocessing:** Handles missing values, encoding categorical variables, etc.
  - **Tree Visualization:** Visualizes the decision tree structure and splits.

### 4. Experiments on Synthetic Data (`tree/classification-exp.py`)
- **Dataset Generation:** Utilized `make_classification` from Scikit-learn to create a synthetic dataset.
- **Model Training:** Trained the decision tree on training data and evaluated on test data.
- **Cross-Validation:** 
  - Conducted 5-fold cross-validation.
  - Used nested cross-validation to tune hyperparameters such as tree depth and minimum samples split.
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall

### 5. Usage and Testing (`tree/usage.py`)
- Verified correctness of implementation through rigorous tests.
- Ensured the model handles edge cases and computes performance metrics accurately.

### 6. Automotive Efficiency Analysis (`tree/auto-efficiency.py` and `auto-efficiency.ipynb`)
- Applied the decision tree to an automotive efficiency dataset.
- Compared the performance with Scikit-learn’s decision tree implementation.
- Demonstrated model evaluation through performance metrics.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DecisionTree-Library.git
   cd DecisionTree-Library
   ```

---

## Usage

1. Import the `DecisionTree` class:
   ```python
   from tree.base import DecisionTree
   ```

2. Train and evaluate the model:
   ```python
   tree = DecisionTree(max_depth=5, min_samples_split=10)
   tree.fit(X_train, y_train)
   predictions = tree.predict(X_test)
   ```

3. Visualize the tree:
   ```python
   from tree.utils import visualize_tree
   visualize_tree(tree)
   ```

---

## Experiments and Results

- **Synthetic Dataset:**
  - Achieved optimal performance metrics after hyperparameter tuning using nested cross-validation.
- **Automotive Efficiency Dataset:**
  - Performance was on par with Scikit-learn’s implementation, demonstrating the robustness of the custom model.

---

## Future Improvements
- Add support for Random Forests.
- Optimize the algorithm for large datasets.
- Extend visualization capabilities for better interpretability.

---

## Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests for improvements or new features.

---
