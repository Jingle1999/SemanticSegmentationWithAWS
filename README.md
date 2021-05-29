# Semantic Segmentation with AWS SageMaker

INTRO

Machine Learning (ML) or in this case Deep Learning (DL) as special case of ML are very magical things, where you’re not programming an algorithm to calculate a result, but you’re programming an algorithm to take a result and to show you patterns within the result. In the case of Semantic Segmentation, you teach the system to visualise objects and put these objects into categories or classes (for different cases of object detection on Amazon Web Services (AWS) SageMaker as ML-tool see the following post: https://medium.com/@kolungade.s/object-detection-image-classification-and-semantic-segmentation-using-aws-sagemaker-e1f768c8f57d). 

As I am working within the Automotive Industry for several years, I am really interested in putting the ideas of ML and DL into the Automotive Industry. Self-Driving cars on the streets, Self-Working robots within a manufacturing process, or Self-Organized Logistics within a production plant are not future thinking anymore, but possible solutions for the Automotive Industry nowadays. Semantic segmentation is the basis for all these tasks. Before you learn how to fly, you need to learn, how to walk. Actually, I start walking now and am already exited, which next steps I’ll make.


PROJECT TASK

Based on a paper of Jonathan Long, Evan Shelhamer, and Trevor Darrell of UC Berkley (https://arxiv.org/pdf/1411.4038.pdf) I will set up Neural Network doing binary segmentation on a part of the Kitti dataset (http://www.cvlibs.net/datasets/kitti/eval_road.php).

![image](https://user-images.githubusercontent.com/65500947/120071690-ba65f600-c090-11eb-8a98-0835345b2577.png)

Fully convolutional network for semantic segmentation


PROCESS FLOW

After setting up the correct AWS environment for ML I loaded the VGG16-model as encoder and starting point of the Neural Network. It takes an input image and generates a high-dimensional feature vector as well as aggregates features at multiple levels. First, we built up the VGG16-model, creating different layers of Neural Network, optimizing these layers, building the correct labels and using our train_nn-function to train different batches of our data-label-combination. The decoder takes a high-dimensional feature vector and generates a semantic segmentation mask. Hence, it decodes features aggregated by the encoder at multiple levels.

For the usage of the code make sure to have AWS set up appropriately and use the following packages:

•	Python 3

•	TensorFlow

•	NumPy

•	SciPy


