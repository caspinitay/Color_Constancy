#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:33:58 2017
 
Classify virtual world images by target object luminance.
 
@author: Nitay Caspi
"""
 
import tensorflow as tf
import numpy as np
import time
 
from build_nn import load_data, split_train_test, feed_forward_deep
from build_nn import calculate_cost, training_step, evaluate_network
 
# The virtual world image dataset has 10 classes
NUM_CLASSES = 10
 
# Virtual world images have dimensions 51 x 51 x 3
IMAGE_DIM = 100
NUM_CHANNELS = 1
 
# Convolution filters have dimensions 10 x 10 x 3
FILTER_DIM = 10
 
def fill_feed_dict(batch_data, batch_labels, x, y_):
    """
    Assign batch data and labels to placeholders for current session.
     
    Arguments:
        batch_data: images for current session
        batch_labels: labels for current sesssion
        x: image placeholder
        y_: labels placeholder
     
    Returns:
        feed_dict: dictionary with assigned placeholders
    """
    feed_dict = {
            x: np.transpose(batch_data),
            y_: np.subtract(batch_labels,1)}
    return feed_dict
 
def run_training(data_train, labels_train, data_test, labels_test, hidden_sizes, logdir, max_steps):
    """
    Central function for training the network. Build the network to feed
    forward, calculate cost, and backpropagate for each training step, repeated
    until cost function converges or max_steps reached. Evaluate network every
    eval_gap steps.
     
    Arguments:
        data_train
        labels_train
        batch_size: size of training data to be used for training
    """
    with tf.Graph().as_default():
         
        # Useful variables
        vectorized_dim = IMAGE_DIM * NUM_CHANNELS
        image_axis = 1
        num_subclasses = 10
        train_size = data_train.shape[1]
                 
        # Labels placeholder same for conv or full-feed
        y_ = tf.placeholder(tf.float32, shape=[1, train_size])
                 
        # Data Placeholder
        x = tf.placeholder(tf.float32, shape= [train_size, vectorized_dim])
                 
        # Build network
        logits = feed_forward_deep(x, hidden_sizes)
        cost = calculate_cost(logits, y_, num_subclasses)
        train_op = training_step(cost)
        (predictions, train_accuracy, rmse) = evaluate_network(logits, y_, num_subclasses)
         
        # Build summary tensor based on summary stats
        summary = tf.summary.merge_all()
         
        # Global variable initializer
        init = tf.global_variables_initializer()
         
        # Saver for writing evaluation files
        saver = tf.train.Saver()
         
        # Session
        sess = tf.Session()
         
        # Summary writer to write summary files in logdir
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
         
        # Begin session by initializing variables
        sess.run(init)
         
        print("BEGIN TRAINING\n")
        start_time = time.time()
        for i in range(max_steps):
                         
            # Fill feed_dict with training batch
            feed_dict = fill_feed_dict(data_train, labels_train, x, y_)
                         
            # Run one training step
            (_, cost_val) = sess.run([train_op, cost], feed_dict=feed_dict)
             
            # Update summary file
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
             
            global test_preds
            # Evaluate network every 10 steps
            if i%10 == 0:
                print("Step %03d, cost %2.5f, time %2.5f" %(i, cost_val, time.time() - start_time))
                (preds, accuracy, rmse_i) = sess.run([predictions, train_accuracy, rmse], feed_dict=feed_dict)
                print("        train accuracy: %2.5f, rmse: %2.5f" %(accuracy, rmse_i))
                if i%100 == 0:
                    test_feed_dict = fill_feed_dict(np.repeat(data_test,9, axis=image_axis), np.repeat(labels_test,9,axis=1), x, y_)
                    test_preds, test_accuracy, test_rmse = sess.run([predictions, train_accuracy, rmse], feed_dict=test_feed_dict)
                    print("        test  accuracy: %2.5f, rmse: %2.5f" %(test_accuracy, test_rmse))
                 
            # Save summary file
            if accuracy > 0.999 and i%100 == 1 or i == max_steps-1:
                test_preds = np.take(test_preds, range(0,train_size,9)) + 1
                checkpoint_file = logdir + '/model.ckpt'
                print checkpoint_file
                saver.save(sess, checkpoint_file)
                break
             
def main(_):
    """
    Call functions to load and sort data, then train network.
    """
         
    # Load Data
    data, labels, indices = load_data('case14b_svd100.csv', 'case14b_svd_labels.csv')
    set_size = data.shape[1]
 
 
    # Split training and testing
    train_prop = 0.9
    global labels_test
    global test_indices
    (data_train, labels_train, data_test, labels_test) = split_train_test(
            data, labels, train_prop)
    test_range = range(int(train_prop*set_size),set_size)
    test_indices = np.take(indices,test_range)
                 
    # Hidden architecture
    large = 1000
    medium = 300
    small = 30
    hidden_sizes = [large, medium, medium, small, small]
     
    # Directory to save summary files
    logdir = '/tmp/tensorflow/logs/case2_svd'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
     
    # Train the network
    run_training(data_train, labels_train, data_test, labels_test, hidden_sizes, logdir, 100000)
     
    # Transpose labels and indices for quicker saving by hand
    labels_test = np.transpose(labels_test)
 
         
# Run tensorflow app
tf.app.run(main=main)