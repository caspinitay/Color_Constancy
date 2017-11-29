#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:50:49 2017
 
Build a convolutional neural network to classify virtual world images by
target object luminance.
 
@author: Nitay Caspi
"""
 
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
 
# Directory with all the datasets
data_dir = '/Users/color_user/Documents/Python/Data/'
sys.path.append(data_dir)
 
# The virtual world image dataset has 10 classes, but model can train on 
# n subclasses
NUM_CLASSES = 10
 
def normalize_and_shuffle(data, labels):
    """
    Normalize cone-response arrays (divide each image by its mean), reshape
    image tensor to correctly feed into convolution, shuffle data and labels
    into random order (maintaining data-label indices)
     
    Arguments:
        data: [sample_size, num_samples]
        labels: [1, num_samples]
         
    Returns:
        processed_data: pre-processed, shuffled [sample_size, num_samples]
        processsed_labels: [1, num_samples]
    """
    print("PRE-PROCESSING DATA...")
     
    num_samples = data.shape[1]
     
    # Normalize images by mean (contrast)
    image_means = np.mean(data, axis=0)
    norm_data = np.divide(data, image_means)
     
    # Shuffle data and labels
    rand_indices = np.random.randint(0, num_samples, size=(1, num_samples))
    rand_data = np.take(norm_data, rand_indices, axis=1)
    rand_labels = np.take(labels, rand_indices, axis=1)
     
    # Remove extra axes
    processed_data = np.squeeze(rand_data, axis=1)
    processed_labels = np.squeeze(rand_labels, axis=0)
         
    return processed_data, processed_labels, rand_indices
 
def split_train_test(data, labels, proportion):
    """
    Splits dataset into training and testing sets based on desired training
    proportion.
     
    Arguments:
        data: pre-processed images [sample_size, num_samples]
        labels: luminance category labels [1, num_samples]
        percent: proportion of original datset devoted to training
         
    Returns:
        data_train: training images
        labels_train: training labels
        data_test: test images
        labels_test: test labels
    """
    print("SPLITTING TRAIN AND TEST SETS...")
         
    # Define index ranges based on train size
    total_samples = data.shape[1]
    num_training = int(proportion * total_samples)
    train_indices = range(num_training)
    test_indices = range(num_training, total_samples)
     
    # Training data and labels
    data_train = np.take(data, train_indices, axis=1)
    labels_train = np.take(labels, train_indices, axis=1)
     
    # Testing data and labels
    data_test = np.take(data, test_indices, axis=1)
    labels_test = np.take(labels, test_indices, axis=1)
     
    return data_train, labels_train, data_test, labels_test
 
def load_data(data_filename, labels_filename):
    """
    Load images and data from respective CSV files if not already in
    local variables and pre-processes data (CSV files found in data_dir)
     
    Arguments:
        data_filename
        labels_filename
     
    Returns:
        data: normalized and shuffled image set [sample_size, num_samples]
        labels: shuffled labels [1, num_samples]
    """
    # Read CSV files
    print("LOADING DATA...")
    raw_data = pd.io.parsers.read_csv(data_filename).as_matrix()
    raw_labels = pd.io.parsers.read_csv(labels_filename).as_matrix()
          
    return normalize_and_shuffle(raw_data, raw_labels)
 
def feed_forward_with_summary(images, hidden_size, keep_prob, num_subclasses):
    """
    Define ops for forward propagation with summary statistics: default 
    architecture consists of one hidden layer with summary statistics for weights,
    biases, and activations. Uncomment layers two and three for deeper
    architecture.
     
    Arguments:
        images: [num_samples, sample_size]
        hidden_size: number of neurons in hidden layer
        keep_prob: proportion of output nodes turned "on" for each training step
        num_subclasses: number of classes if less than 10 (default 10)
         
    Returns:
        logits: output layer activations [num_samples, num_subclasses]
         
    Note:
        For changes in architecture, update variable 'h_hidden_drop' to input
        activations from final hidden layer.
    """
     
    print "BUILDING GRAPH... "
     
    sample_size = int(images.shape[1])
    num_samples = int(images.shape[0])
 
    # Construct a tensorflow variable for weight parameters
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
     
    # Construct a tensorflow variable for bias parameters
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
     
    with tf.name_scope('hidden1'):
        hidden_weights1 = weight_variable([sample_size, hidden_size])
        hidden_biases1 = bias_variable([hidden_size])
        raw_activations1 = tf.matmul(images, hidden_weights1) + hidden_biases1
        h_hidden1 = tf.nn.relu(raw_activations1)
         
        # Hidden layer summary stats
        tf.summary.histogram('raw_activations_dist', raw_activations1)
        tf.summary.image('raw_activations_visual', tf.reshape(
                raw_activations1, [1, num_samples, hidden_size, 1]))
        tf.summary.histogram('rectified_activations', h_hidden1)
        tf.summary.histogram('hidden_weights1', hidden_weights1)
        tf.summary.image('hidden_weights_visual', tf.reshape(
                hidden_weights1, [1, sample_size, hidden_size, 1]))
        tf.summary.histogram('hidden_biases1', hidden_biases1)
         
    with tf.name_scope('hidden2'):
        hidden_weights2 = weight_variable([hidden_size, hidden_size])
        hidden_biases2 = bias_variable([hidden_size])
        raw_activations2 = tf.matmul(h_hidden1, hidden_weights2) + hidden_biases2
        h_hidden2 = tf.nn.relu(raw_activations2)
        
        # Hidden layer summary stats
        tf.summary.histogram('raw_activations_dist', raw_activations2)
        tf.summary.image('raw_activations_visual', tf.reshape(
                raw_activations2, [1, num_images, hidden_size, 1]))
        tf.summary.histogram('rectified_activations', h_hidden2)
        tf.summary.histogram('hidden_weights2', hidden_weights2)
        tf.summary.image('hidden_weights_visual', tf.reshape(
                hidden_weights2, [1, hidden_size, hidden_size, 1]))
        tf.summary.histogram('hidden_biases2', hidden_biases2)    
    with tf.name_scope('hidden3'):
        hidden_weights3 = weight_variable([hidden_size, hidden_size])
        hidden_biases3 = bias_variable([hidden_size])
        raw_activations3 = tf.matmul(h_hidden2, hidden_weights3) + hidden_biases3
        h_hidden3 = tf.nn.relu(raw_activations3)
        
        # Hidden layer summary stats
        tf.summary.histogram('raw_activations_dist', raw_activations3)
        tf.summary.image('raw_activations_visual', tf.reshape(
                raw_activations3, [1, num_images, hidden_size, 1]))
        tf.summary.histogram('rectified_activations', h_hidden3)
        tf.summary.histogram('hidden_weights', hidden_weights3)
        tf.summary.image('hidden_weights_visual', tf.reshape(
                hidden_weights3, [1, hidden_size, hidden_size, 1]))
        tf.summary.histogram('hidden_biases3', hidden_biases3)
 
    with tf.name_scope('output'):
        h_hidden_drop = tf.nn.dropout(h_hidden1, keep_prob)
        output_weights = weight_variable([hidden_size, num_subclasses])
        output_biases = bias_variable([num_subclasses])
        logits = tf.matmul(h_hidden_drop, output_weights) + output_biases
         
        # Output layer summary stats
        tf.summary.histogram('logits_dist', logits)
        tf.summary.image('logits_visual', tf.reshape(
                logits, [1, num_samples, num_subclasses, 1]))
        tf.summary.histogram('output_weights', output_weights)
        tf.summary.image('output_weights_visual', tf.reshape(
                output_weights, [1, hidden_size, num_subclasses, 1]))
        tf.summary.histogram('output_biases', output_biases)
     
    return logits
 
def feed_forward_deep(images, hidden):
    """
    Define ops for forward propagation with deep architectures (exclude summary
    statistics for speed). Hidden layer sizes arranged in "size" array, and the
    number of hidden layers equals the length of the size array.
     
    Arguments:
        images: [num_samples, sample_size]
        hidden_size:
     
    Returns:
        logits [num_samples, num_subclasses]
     
    Note:
        For changes in architecture, update variable 'h_hidden_drop' to input
        activations from final hidden layer.
    """
     
    print "BUILDING GRAPH... "
     
    sample_size = int(images.shape[1])
    num_hidden = len(hidden)
     
    # Construct a tensorflow variable for weight parameters
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
     
    # Construct a tensorflow variable for bias parameters
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
     
    with tf.name_scope('hidden1'):
        hidden_weights = weight_variable([sample_size, hidden[0]])
        hidden_biases = bias_variable([hidden[0]])
        raw_activations = tf.matmul(images, hidden_weights) + hidden_biases
        h_hidden = tf.nn.relu(raw_activations)
        print "    hidden 1 size: ", hidden[0]
         
    with tf.name_scope('hidden2'):
        hidden2_weights = weight_variable([hidden[0], hidden[1]])
        hidden2_biases = bias_variable([hidden[1]])
        h_hidden2 = tf.nn.relu(tf.matmul(h_hidden, hidden2_weights) + hidden2_biases)
        print "    hidden 2 size: ", hidden[1]
         
    with tf.name_scope('hidden3'):
        hidden3_weights = weight_variable([hidden[1], hidden[2]])
        hidden3_biases = bias_variable([hidden[2]])
        h_hidden3 = tf.nn.relu(tf.matmul(h_hidden2, hidden3_weights) + hidden3_biases)
        print "    hidden 3 size: ", hidden[2]
     
    with tf.name_scope('hidden4'):
        hidden4_weights = weight_variable([hidden[2], hidden[3]])
        hidden4_biases = bias_variable([hidden[3]])
        h_hidden4 = tf.nn.relu(tf.matmul(h_hidden3, hidden4_weights) + hidden4_biases)
        print "    hidden 4 size: ", hidden[3]
     
    with tf.name_scope('hidden5'):
        hidden5_weights = weight_variable([hidden[3], hidden[4]])
        hidden5_biases = bias_variable([hidden[4]])
        h_hidden5 = tf.nn.relu(tf.matmul(h_hidden4, hidden5_weights) + hidden5_biases)
        print "    hidden 5 size: ", hidden[4]
 
    with tf.name_scope('output'):
        output_weights = weight_variable([hidden[num_hidden-1], NUM_CLASSES])
        output_biases = bias_variable([NUM_CLASSES])
        logits = tf.matmul(h_hidden5, output_weights) + output_biases
     
    return logits
 
def calculate_cost(logits, labels, num_subclasses):
    """
    Given output from forward propagation, compare logits and labels to calculate
    loss for training iteration.
     
    Arguments:
        logits: softmax output from feed_forward [num_samples, num_subclasses]
        labels: one-hot labels vector for batch images [1, num_samples]
        num_subclasses
     
    Returns:
        cost: cross-entropy cost tensor, callable tf op
    """
    # Shift labels to fit subclasses
    labels = tf.ceil(tf.multiply(tf.divide(labels,10),num_subclasses))
         
    with tf.name_scope('cost'):
        oh_labels = tf.transpose(tf.one_hot(tf.cast(labels, tf.int64), 
                            num_subclasses, on_value=1, off_value=0, axis=0))
        oh_labels = tf.squeeze(oh_labels) # one-hot labels tensor
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=oh_labels, logits=logits), name='cross_entropy')
        tf.summary.scalar('cost', cost)
         
    return cost
 
def training_step(cost):
    """
    Backpropagation op to update learning parameters. Uses AdamOptimizer to
    automatically update learning rate.
     
    Arguments:
        cost
         
    Returns:
        train_op: callable tf op to initiate training
    """
    train_op = tf.train.AdamOptimizer().minimize(cost)
         
    return train_op
 
def calculate_rmse(predictions, labels, num_subclasses):
    """
    Calculate the root mean square error of the estimated values from a forward
    propagaion.
     
    Arguments:
        predictions: [num_samples, num_subclasses]
        labels: [1, num_subclasses]
        num_subclasses
     
    Returns:
        rmse
    """     
    # Alternative RMSE implementation
    labels = tf.cast(labels+1, tf.int64)
    diffs = tf.subtract(predictions+1, labels) # predictions = argmax(logits)
    scaled_diffs = tf.divide(diffs, labels)
    squared_diffs = tf.square(scaled_diffs)
    rmse = tf.sqrt(tf.reduce_mean(squared_diffs))
     
    return rmse
 
def evaluate_network(logits, labels, num_subclasses):
    """
    Evaluate the network's accuracy against periodic training set or testing
    set. Accuracy defined as proportion of correct classifications. Calculate
    RMSE for model prediction versus true value.
     
    Arguments:
        logits [num_samples, num_subclasses]
        labels [num_samples, num_subclasses]
        num_subclasses
         
    Returns:
        accuracy: number of correct classifications divided by data size
        rmse: root mean squared error of model prediction versus label value
    """       
    # Accuracy
    with tf.name_scope('accuracy'):
        labels = tf.cast(labels, tf.int64)
        predictions = tf.argmax(logits,1)
        correct_prediction = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        differences = tf.abs(tf.subtract(predictions, labels))
        tf.summary.histogram('differences', differences)
        tf.summary.scalar('accuracy', accuracy)
     
    # RMSE
    with tf.name_scope('rmse'):
        rmse = calculate_rmse(predictions, labels, num_subclasses)
        tf.summary.scalar('rmse', rmse)
     
    return predictions, accuracy, rmse
 