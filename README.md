# Image Segmentation Using Mask R-CNN on COCO 2017 Dataset

## Overview
This project focuses on performing **image segmentation** using the **Mask R-CNN** architecture. The goal is to segment images into specific categories such as **cake**, **car**, **dog**, and **person**. The project uses the **COCO 2017 dataset** for training, validation, and testing the segmentation model.

## Objective
The objective is to train a model using **Mask R-CNN**, a state-of-the-art deep learning model, to perform image segmentation. This involves detecting and segmenting objects in images based on the categories provided. Mask R-CNN not only predicts bounding boxes but also generates segmentation masks for each object, allowing for more precise localization and classification.

## Dataset
The dataset used in this project is the **COCO 2017 dataset**, which contains labeled images for various categories. The dataset was filtered to focus on the following categories:
- **Cake**
- **Car**
- **Dog**
- **Person**

The dataset was split into:
- **Training Set**: 300 images
- **Validation Set**: 300 images
- **Test Set**: 30 images

## Methodology

### Data Preprocessing
- **COCO Dataset** was loaded using the **COCO API**, which helped in parsing image annotations and categories.
- Images were resized and normalized for training.
- The images were augmented using transformations such as random flipping and cropping to improve model generalization.

### Model Architecture
- **Mask R-CNN**: The model uses the **ResNet-50 backbone** with a Feature Pyramid Network (FPN) for generating high-resolution object segmentation masks. Mask R-CNN is an extension of Faster R-CNN and adds a segmentation mask output for each detected object.
- **Training Hyperparameters**: The model was trained with the following settings:
  - **Learning Rate**: 0.001
  - **Batch Size**: 2
  - **Optimizer**: SGD (Stochastic Gradient Descent)
  - **Epochs**: 10

### Training Process
- The model was trained on a subset of the COCO dataset using the Mask R-CNN architecture.
- **Data Augmentation** techniques were used to enhance the model's ability to generalize.
- The model was evaluated using **IoU (Intersection over Union)** and **AP (Average Precision)** metrics for assessing segmentation performance.

### Inference and Visualization
- After training, the model was tested on a set of images to predict the segmentation masks.
- The predicted masks were overlaid on the original images for visual inspection.
- Results were displayed using **Matplotlib** to show segmented objects with bounding boxes and masks.

## Key Findings

- **Model Accuracy**: The Mask R-CNN model performed well on images with clear object boundaries. The average **IoU** score was high, indicating accurate object localization.
- **Challenges**: The model faced difficulty when objects were occluded or tightly packed, leading to less accurate segmentation in some test cases.
- **Visualization**: The model successfully segmented and labeled objects in test images, with predictions displayed as bounding boxes and segmentation masks.

## Results
- The Mask R-CNN model successfully segmented **cake**, **car**, **dog**, and **person** categories with high accuracy on well-defined objects.
- **Future Improvements**: Additional fine-tuning and the use of more advanced augmentation techniques may improve performance on occluded or complex scenes.

## Limitations
- The dataset was limited in size (300 training images), which might affect the generalization of the model to larger, more diverse datasets.
- **Occlusions and Clutter**: The model performed poorly when objects were partially occluded or packed tightly in the images.

## Future Work
- **Model Fine-tuning**: Further fine-tuning of the model with longer training and more advanced techniques such as **Non-Maximum Suppression (NMS)** could help reduce overlapping boxes.
- **Data Expansion**: Using a larger dataset with more diverse categories may improve the robustness of the model.
- **Other Transformer Models**: Experimenting with other models like **DETR** (Detection Transformer) for segmentation might also yield better results.

## Conclusion
This project demonstrates the effectiveness of **Mask R-CNN** for image segmentation tasks. By fine-tuning the model and using various data augmentation techniques, the model is able to segment objects accurately in test images, although there are some challenges with occlusions and overlapping objects.

## References
- **Mask R-CNN**: He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2960-2968.
- **COCO Dataset**: Lin, T. Y., Ma, L., & Belongie, S. (2014). Microsoft COCO: Common Objects in Context. Proceedings of the European Conference on Computer Vision (ECCV).
