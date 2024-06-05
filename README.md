# Questions

## Complete decision tree implementation in tree/base.py. 

In tackling the task of implementing a decision tree in the tree/base.py file, I focused on ensuring that the decision tree works across four different cases: i) discrete features with discrete output, ii) discrete features with real output,iii) real features with discrete output, and iv)real features with real output. Here's an overview of the steps I took to complete this task:

## Implementation in tree/base.py

### Decision Tree Class
1. Initialization: I set up the initialization method to accept parameters like the criterion for splitting, the maximum depth of the tree, and the minimum samples required to split a node.
2. Fitting the Tree: Implemented the fit method to build the tree recursively. This method uses the chosen splitting criterion (Gini Index, Information Gain for discrete output, or Information Gain based on MSE for real output) to decide the best splits.
3. Prediction: Added a predict method to traverse the built tree and make predictions for new data points.
4. Splitting: Created methods to find the best split and to split the data based on features and threshold values.
5. Stopping Criteria: Incorporated stopping criteria based on maximum depth and minimum sample split to prevent overfitting.
### Metrics in metrics.py
1. Performance Metrics Functions: Completed functions for calculating Gini Index, Entropy (for Information Gain), and MSE (Mean Squared Error) to be used as criteria for splitting the nodes in the decision tree.
2. Helper Functions: Added helper functions to compute the necessary metrics given subsets of the data.
### Utilities in utils.py
1. Data Handling Functions: Implemented functions to handle data preprocessing tasks, such as splitting data into training and testing sets and normalizing or binarizing features as needed.
2. Tree Visualization: Added functions to visualize the tree, either textually or graphically, to help with understanding the structure and decisions made by the tree.
### Running and Testing with usage.py
1. Testing: Ran the provided `usage.py` script to verify the correctness of the implementation. This involved:
    - Checking if the decision tree can handle all four cases (discrete features/discrete output, discrete features/real output, real features/discrete output, real features/real output).
    - Ensuring that the performance metrics are computed correctly and used appropriately during tree building.
    - Verifying the accuracy of predictions and the overall functionality of the tree.


### File Structure
- `metrics.py`: Complete the performance metrics functions in this file. 

- `usage.py`: Run this file to check your solutions.

- tree (Directory): Module for decision tree.
    - `base.py` : Complete Decision Tree Class.
    - `utils.py`: Complete all utility functions.

> You should run `usage.py` to check your solutions. 


## Part 2: Decision Tree Experiments 

    Generate your dataset using the following lines of code

    ```python
    from sklearn.datasets import make_classification
    X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

    # For plotting
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    ```

Implemented the decision tree model on the above datasets and performed the following experiments:

a) The first 70% of the data used for training purposes and the remaining 30% for test purposes. Trained the model on training data and found the accuracy, per-class precision and recall of the decision tree model on the test data.

b) Then used 5 fold cross-validation on the dataset. Using nested cross-validation found the optimum depth of the tree. 

>  See `classification-exp.py` and `classification.ipynb` files to look at the code for the above experiments.

## Part 3 
a) Shown the usage of our decision tree for the [automotive efficiency](https://archive.ics.uci.edu/ml/datasets/auto+mpg) problem.

b) Compared the performance of your model with the decision tree module from scikit learn.
    
   > Have a look at `auto-efficiency.py` and `auto-efficiency.ipynb` files for the code and results for the above task.
    
