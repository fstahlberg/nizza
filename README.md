# Neural word alignment
Neural versions of SMT-style word alignment models like the IBM model hierarchy
or HMM-based alignment models.

This project is based on TensorFlow’s high-level machine learning API 
(tf.estimator).

# Getting started
This repository contains implementations of neural alignment models, and the 
infrastructure for training and basic inference. We do not provide functionality
for data generation, preprocessing, tokenization etc. Instead, we maintain 
compatibility with the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
project, ie. training and dev sets created by T2T data generators can be used
in Nizza.

*Author: Felix Stahlberg*
