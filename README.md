# Optimizing Antenna Array Design using Deep Learning

## Abstract

The deployment of antenna arrays in wireless communication systems has witnessed significant advancements in recent years. A well-designed antenna array can enhance the power radiated in the desired direction, achieving a higher gain than a single antenna. However, if the design is flawed, the antennas can interfere with each other. The goal of the project is to utilize deep learning to improve a given pattern.

## 1. Introduction

Optimizing antenna arrays to achieve desired patterns is a formidable challenge in the realm of wireless communication. Particularly, the application of deep learning techniques to tackle this problem remains relatively unexplored. This research paper aims to delve into this uncharted territory by investigating the potential of deep learning in addressing the complexities of antenna array optimization. The study focuses on two crucial constraints: the need to maintain a constant number of antennas and the requirement for a minimum spacing between them. By exploring these aspects, we aim to shed light on the untapped potential of deep learning for optimizing antenna arrays and contributing to advancements in the field of wireless communication.

Designing an antenna array with a specific pattern is difficult due to the intricate relationship between array geometry and radiation patterns. Additionally, the constraints of fixed antenna numbers and minimum spacing further complicate the optimization process.

To tackle this challenge, we propose a two-part approach. Firstly, we develop a deep learning predictive model to estimate the cost associated with different antenna array designs. This predictive model is trained using a dataset of known configurations and associated costs. Secondly, leveraging the insights gained from the predictive model, we employ optimization algorithms to improve the inputs and minimize the cost.

## 2. Datasets

The dataset contains a total of 100,000 patterns, each consisting of a variable number of antennas ranging from 800 to 1024, where Y and Z range from -33 to 33. The dataset provides crucial information for each pattern, including the corresponding cost, which is a scalar value indicating the quality of the pattern. The cost is calculated by the function F, producing a low value if the pattern is good and a high value if the pattern has more interference. Additionally, each antenna within a pattern is characterized by its Y and Z coordinates, representing the spatial positioning.

The range of the cost values within the dataset spans from -0.07 to -40,000. The mean cost is approximately -10,000, with a standard deviation of around 9,000. These statistics indicate a considerable variation in the costs, highlighting the diverse quality of the antenna array patterns.

## 3 Methods

### 3.1 Predicting the cost

The initial stage of this research project focuses on approximating the cost function F, which plays a pivotal role in evaluating the performance of antenna array patterns. Our primary objective is to construct a robust model that can accurately predict the cost associated with various antenna array designs. To accomplish this goal, we explore the effectiveness of two distinct models: a feedforward neural network (FNN) and a set transformer.

For training purposes, we employ the Adam optimizer with a learning rate of 1e-3. The mean squared error (MSE) is utilized as the loss function, enabling us to measure the disparity between the predicted costs and the actual costs.

#### 3.1.1 Feedforward Neural Network (FNN)

For the FNN, we can use both YZC-scale and C-scale without much difference. The model takes in a 1-D vector arranged as $Y_1, Y_2, ..., Y_{1024},Z_1,...,Z_{1024}$, followed by two hidden layers with 8 neurons and ReLU activation to produce the cost.

### 3.2 Optimizing the inputs

In this section, we utilize the FNN, represented as G, as an approximation of the cost function. The objective is to examine how the cost changes with variations in the input by calculating the gradient of G with respect to the inputs. By leveraging the gradient and employing a learning rate, we can employ gradient descent to optimize the input space.

During the optimization process, certain considerations are taken into account to ensure the validity and feasibility of the antenna array designs. First, we apply a padding mask to maintain a consistent number of antennas throughout the optimization. This step helps maintain the structural integrity of the array.

Additionally, a constraint check is performed to enforce a minimum distance between antennas. This constraint aims to prevent antennas from being positioned too close to each other, which may lead to interference or degradation in performance. The minimum distance is set to 0.5 times the Euclidean distance between the antennas. If this constraint is violated during the optimization, the gradient step is not accepted, but the optimization process continues to explore other possibilities until a set number of iterations.