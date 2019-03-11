# AdaHMG
AdaHMG: A new updating rule is proposed based on mixed high power historical and current squared gradients to construct a targeted first-order optimization algorithm for the adaptive learning rate. Both the loss function curve and the evaluation metrics show that AdaHMG algorithm has better performance in accuracy, RMSE, and convergence (Especially suitable for time series prediction tasks!!).
# References
Jun Hu and Wendong Zheng, "An Adaptive Optimization Algorithm Based on Hybrid Power and Multidimensional Update Strategy", IEEE Access, 2019, vol. 7(1), pp. 19355-19369.
# Datasets
1. PM2.5 Air Quality Forecasting. This dataset is the related air data which is obtained from UCI and collected near the US Embassy in China during 2010 to 2014. The preprocessing of data is similar with the previous experiment. The data from the first 1.5 years serve as the training set and the rest as a testing set (or validation set).
2. Handwritten digit recognition task (MNIST). Convolution neural networks are used to process classical MNIST datasets. A total of 60,000 records in the training set are transformed into 60,000 images of 28 * 28 * 1 pixels (length, width and monochrome), respectively. Thus, a 4D matrix of 60,000 * 28 * 28 * 1 is generated and then the pixel values are standardized. The dataset is divided into a training set and a validation set as 4:1, and the testing set is an additional 10,000 pieces of data. Finally, the labels of the training and test dataset are one hot encoded. The model includes two layers of convolution and pooling. The hidden layer is fully connected with 128 neurons, and the dropout function is added to avoid the overfitting. There are 10 neurons in the output layer.
# Requirements
1. You will need to copy the optimizers.py file to this directory(e.g. G:\Users\Anaconda3\envs\tensorflow\Lib\site-packages\keras) to replace the previous files.
2. TensorFlow 1.4-gpu
3. python 3.5
4. Keras 2.1.5
5. numpy
6. matplotlib
7. pandas
8. sklearn
# Usage
1. train AdaHMG algorithm on PM2.5 dataset:
python LSTM_pollution1.py
2. train AdaHMG algorithm on MNIST dataset:
python mnist-cnn.py
# Our Experiment Environment
The CPU of the experimental device is i7-7700HQ, and the GPU is NVIDIA's GTX 1050 with a memory size of 2GB.
# Convergence Analysis
Detailed theoretical analysis is in Section 4 and in the appendix of the AdaHMG-access-01561.pdf.
