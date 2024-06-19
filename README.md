# Brain Tumor Segmentation Using Hybrid Inception-ResNet Based UNet Architecture

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Dataset](#dataset)
   - [Preprocessing](#preprocessing)
   - [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Implementation Details](#implementation-details)
5. [Results](#results)
   - [Quantitative Results](#quantitative-results)

## Introduction

Brain tumor segmentation from magnetic resonance imaging (MRI) images is a critical task in medical image analysis. Accurate segmentation facilitates timely diagnosis and effective treatment planning for brain tumors. The complexity of brain structures and the variability of tumor appearance in MRI images pose significant challenges for automated segmentation methods. Traditional image processing techniques often fall short in accurately delineating tumor boundaries due to the intricate and heterogeneous nature of brain tumors.

To address these challenges, we implemented a hybrid model that combines elements of Inception modules, Residual Networks (ResNet), and the UNet architecture. This approach leverages the strengths of each component to achieve more precise segmentation results. Specifically, the Inception modules help capture multi-scale features, ResNet blocks aid in effective gradient propagation, and the UNet structure ensures detailed localization with its encoder-decoder framework.

## Methodology

### Dataset

The Brain Tumor Segmentation Challenge 2020 (BraTS 2020) dataset was used in this project. This dataset provides a comprehensive set of multi-modal MRI images, including T1, T2, FLAIR, and T1Gd sequences. Each image is annotated with labels for different tumor regions: enhancing tumor, whole tumor, and tumor core. The dataset is designed to present a wide variety of tumor appearances, making it a robust benchmark for evaluating segmentation algorithms.

### Preprocessing

1. **Normalization**: Each MRI slice was normalized to have zero mean and unit variance. This step is crucial to ensure that the intensity values are on a comparable scale, which aids in the training of the neural network.
2. **Resizing**: The MRI slices were resized to a consistent size (e.g., 128x128 or 256x256 pixels) to fit the input requirements of our model. Resizing helps in maintaining a uniform input dimension, simplifying the model architecture design.
3. **Data Augmentation**: To increase the diversity of the training data and prevent overfitting, data augmentation techniques such as rotation, flipping, scaling, and elastic deformations were applied. These augmentations simulate variations in the dataset, helping the model generalize better to unseen data.

### Model Architecture

#### Baseline 2D U-Net

The U-Net model is widely used for biomedical image segmentation due to its ability to capture both spatial and contextual information effectively. The U-Net consists of an encoder-decoder structure with skip connections.

1. **Encoder**: The encoder captures the context of the input image. It consists of repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for down-sampling. At each down-sampling step, the number of feature channels is doubled.
2. **Decoder**: The decoder reconstructs the segmentation map. It consists of up-sampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels. This is followed by a concatenation with the corresponding cropped feature map from the encoder and two 3x3 convolutions, each followed by a ReLU.
3. **Skip Connections**: Skip connections are used between corresponding layers in the encoder and decoder to retain high-resolution features, which helps in accurate localization.

#### Proposed Hybrid Model

Our proposed hybrid model integrates Inception modules and Residual Network (ResNet) blocks into the UNet architecture to enhance its feature extraction capabilities and gradient flow.

1. **Inception Modules**: In the encoder part, Inception modules are used to capture multi-scale features. Each Inception module consists of multiple convolutions with different filter sizes (e.g., 1x1, 3x3, and 5x5) applied in parallel, and their outputs are concatenated. This design allows the network to capture features at various scales, improving its ability to recognize complex patterns in the MRI images.
2. **Residual Blocks**: In the decoder part, Residual blocks are employed to facilitate better gradient flow and faster convergence during training. Each Residual block consists of two or more convolutional layers with skip connections that bypass one or more layers. These connections help in mitigating the vanishing gradient problem and allow the model to learn deeper representations.

## Training

1. **Loss Function**: We used a combination of Dice Loss and Binary Cross-Entropy Loss to handle class imbalance and ensure better segmentation performance. Dice Loss focuses on the overlap between the predicted and ground truth segments, while Binary Cross-Entropy Loss penalizes incorrect classifications.
2. **Optimizer**: The Adam optimizer was used with an initial learning rate of 0.001. Adam combines the advantages of AdaGrad and RMSProp, providing an adaptive learning rate for each parameter and ensuring faster convergence.
3. **Metrics**: Mean Intersection over Union (IoU) and Dice Similarity Coefficient (DSC) were used to evaluate the model's performance. IoU measures the overlap between the predicted and ground truth segments, while DSC is a measure of set similarity that accounts for both true positives and false negatives.

## Implementation Details

1. **Framework**: The model was implemented using TensorFlow and Keras, which provide flexible and powerful tools for designing and training deep learning models.
2. **Hardware**: Training was performed on a GPU to expedite the process. GPUs are well-suited for the parallelizable nature of neural network operations, significantly reducing training time.

## Results

The proposed hybrid model showed improved performance compared to the baseline U-Net model. The use of Inception modules in the encoder helped capture multi-scale features effectively, while the Residual blocks in the decoder improved the model's ability to reconstruct accurate segmentation maps.

### Quantitative Results

| Model                      | Mean IoU | Dice Similarity Coefficient |
|----------------------------|----------|-----------------------------|
| Baseline U-Net             | 0.75     | 0.77                        |
| Hybrid Inception-ResNet U-Net | 0.82     | 0.84                        |
