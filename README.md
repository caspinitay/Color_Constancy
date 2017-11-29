# Color_Constancy

Code samples from summer research project:

1. **preprocess.matlab**:
MATLAB function to pre-process data for deep learning. Reads in images as matrices from .mat file, then compresses images with Singular Value Decomposition algorithm, and saves resulting matrices to .csv file.

2. **build_nn.py**: Python functions used to build Neural Network.
    * *normalize_and_shuffle*: Applies contrast normalization to each image and randomly shuffles image order.
    * *split_train_test*: Splits dataset into training and testing sets based on desired training proportion.
    * *load_data*: Load images and data from respective CSV files if not already in local variables and pre-processes data (CSV files found in data_dir).
    * *feed_forward_with_summary*: Define ops for forward propagation with summary statistics: default  architecture consists of one hidden layer with summary statistics for weights, biases, and activations. Uncomment layers two and three for deeper architecture.
    * *feed_forward_deep*: Define ops for forward propagation with deep architectures (exclude summary statistics for speed).
    * *calculate_cost*: Given output from forward propagation, compare logits and labels to calculate loss for training iteration.
    * *training_step*: Backpropagation op to update learning parameters. Uses AdamOptimizer to automatically update learning rate.
    * *calculate_rmse*: Calculate the root mean square error of the estimated values from a forward propagation.
    * *evaluate_network*: Evaluate the network's accuracy against periodic training set or testing set. Accuracy defined as proportion of correct classifications. Calculate RMSE for model prediction versus true value.
    
3. **luminance_classifier.py**: Calls build_nn functions to read in data, build a neural network, run a training epoch on desired dataset, and test and evaluate network performance.
