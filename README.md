# Dynamic-Backdoor
This repository contains the PyTorch implementation for the EuroS&P 2022 paper "Dynamic Backdoor Attacks Against Machine Learning Models" paper.

You can check the implementation of the CIFAR10 task in `CIFAR_CondGAN.py`, it contains all functions and basic workflow required for the attack.  Note that the code is currently based on PyTorch 0.4.0, with plans for an update to a more advanced version in the near future.

## Main Functions

`convertToOneHotEncoding`: Converts class labels to one-hot encoding.

Input: Class labels (c) and the total number of classes (numOfClasses).
Output: One-hot encoded representation of class labels.

`transformImg`: Applies image transformations using normalization.

Input: Images (image) and an optional scaling factor (scale).
Output: Transformed images.

`insertSingleBD`: Inserts a single backdoor pattern into images for a specified class.

Input: Images (image), backdoor pattern (BD), class label (label), and an optional scaling factor (scale).
Output: Images with the inserted backdoor pattern.

`train`: Trains the main model and backdoor injection model.

Input: Model (model), device, training data loader (train_loader), optimizers (optimizer and optimizerBD), current epoch (epoch), and backdoor model (bdModel).
Output: Updates model parameters based on the loss calculated during training.
