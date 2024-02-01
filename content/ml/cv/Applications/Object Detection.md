---
title: Object Detection
---

Popular Datasets
- PASCAL
- ImageNetf
- COCO
- Open Images
# Traditional Approaches


Problem statement:
- output class scores + box coordinates
### **Traditional Sliding Window**
- Objet detetion naturally adds to top of image classification
- If have an image classifier, simple way to detect objects
	- Slide window aross image, classify whether cropped image in window is of desired type
- Problems
	- Computationally infeasible (size, aspect ratio)
	- Imbalanced
- [ ] Template matching (slide template over image at multple positions.)

#### **Haar-Like Features**
- To detect a given instance (tempalte), a similarity score may be goo neough
- But- to get objet of given class, need
	- Strong features
	- Good classifier
- Haar features - easy to find edges of lines in image
- AdaBoost filters out irrelevant Haar features from a specific object

#### **Histogram of oriented gradients**
- Sliding window at al position / scales but using HOG features instead of previous Haar-like
- After getting feture vector for given image, SVm trained to detect whether feature vector is object of interest or not

Comparison
- Haar: object shading (thus good at frontal face detection)
- HoG: good at object shape (good at pedestrian detection)

#### **Deformable Part model**
- Capures HOG features at two resolutions 




### **Region Proposal**
How to determine most likely regions to contain object non-background?

#### **Selective search**
- Hierarchical segmentation at all scales
- high reacll
- Find “blobby” image regions likely to contain objects

#### **Edge Boxes**
- Scoring candidate box solely on number of edges it wholely encoloses
- This creates a surprisingly effective object proposal mesaure

#### **Non maximum supression**
Given predictions of each class independently, ranked by decreaseing conidence
For i = 2, 3, .. reject prediction $r_i$ if it has **intersection-over-union** overlap higher than a threshold $J(r_i,r_j) > \tau$ 

**Region overlap**
- Given predictions A, B represented as planar point sets
- Intersection-over-union or Jaccard index is $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$

**Problem of NMS**
- Overlap threshold balances 2 conflicting needs
	- If overlapping threshold is larger, proposal less likeley to be supress → larger number of false-positives
	- Smller overalpping threshold → supressed proposals too aggressively
- Recent attemps to improve NMS …..
- Always assume NMS last post-processing stage after each detector

### Detection Metrics
**Detection Evaluation**
- Object Detection (OD) models evaluated using Intersection over Union (IoU) at different thresholds, each $\tau$ can yield varied predictions.
- In each image and for each class, predicted regions are sorted by descending confidence.
- Assign region $r$ to ground truth $g*$ (max overlap) if overlap $J(r,g*) > \tau$; labeled True Positive if so, else False Positive.
- Each ground truth region is matched with at most one predicted region.
- For every class, sort predicted regions by descending confidence, then compute Average Precision (AP) based on true/false labels.
- Mean Average Precision (mAP) is computed as the mean of AP values across all classes.

**Average Precision**
- Represents average precision across all recall levels, summarizing precision-recall curve into a single value.
- Involves creating a ranked list of items (regions) with labels of true/false based on their relevancy.


# Deep Learning Approaches
**Conceptually**

**Classification + Localization**
- Multi-task loss- predict single class + bounding box
- Thus
	- Localization as a regression problem

Localization as regression
- regressing to correct coordinates

Detection as classification
- Sliding window: apply CNN to many different crops of image

Convolutional Sliding window
- Inhrently efficient at sliding winow

Features computed at each scale
- Image pyarmind techique
	- Computaionally intensive

Anchors: predefined bounding boxes in different proprtions
- Use anchors to hanle the scale an aspect ratio problem
- Network must predict location, category, and IoU for every tiled anchor box


**2 mainstream approaches to OD** 
- **1 stage**:
	- Directly use cnn to predict classes + regress boxes
- **2 stages**: 
	- Predict class agnostic regions
	- Classify + regress location of boxes
# Single Stage Approaches
Directly oes image taasks without any region proposal 
Fast.
- YOLO
- SSD
- RetinaNet
- DETR


### YOLO (You Only Look Once)
- First detection model to combine bounding box prediction and class identification in an end-to-end network.
- The YOLO family has evolved since 2016 with versions like YOLOv1, v2, v3, etc.

Rather than treat regions independently, make all predictions all at once
- Divide input image into coarse grid
	- Make a set of base bounding boxes centered at each grid cell
- For each of grid cells’ bounding boxes, predict offset + confidence for each class

Just go from one input image to tensor of scores with one big CNN

### YOLOv1
- Feature extractor: DarkNet architecture with 24 convolutional layers followed by 2 fully connected layers.
- Divides the input image into a 7×7 grid, with each cell responsible for predicting objects whose center falls within it.
- Each cell predicts 2 anchors (B=2) and probabilities for 20 categories (C=20).
- The network computes a product between objectness confidence and class probabilities for each cell.

### YOLOv2 (2017)
- Improvements over YOLOv1: Uses DarkNet-19, replaces fully connected layers with convolutions, incorporates Batch Normalization.
- Uses a 13×13 grid for finer object detail prediction, with 5 anchor boxes (B=5).
- Enhances performance in terms of speed and localization, especially for smaller objects.

### YOLOv3 (2018)
- Builds upon YOLOv2, incorporating DarkNet-53 as the backbone.
- Makes predictions at three hierarchical levels, improving detection of smaller objects.
- Utilizes 3 anchor boxes (B=3) for each scale, enhancing the ability to detect objects of different sizes.

- YOLO models are known for their speed but initially had challenges in accurately localizing smaller objects.
- Each version introduces improvements in feature extraction, grid resolution, and anchor box handling, enhancing the model's accuracy and versatility.



### SSD (Single Shot Detector)
- Utilizes pyramidal feature hierarchy for multi-scale object detection.
- Employs feature maps from different layers of a VGG-16 network.
- Early layers detect small objects, later layers detect larger objects.
- Includes anchor boxes for each cell of a feature map, with a specific number of anchors (e.g., 4 or 6).
- Predicts offsets from anchor box centers rather than direct bounding box coordinates, unlike YOLO.
- Example of anchor box scaling up at conv9 and conv10 layers.
- Pyramid starts at low resolution, presenting challenges with small objects.

### RetinaNet
- Effective one-stage model, especially for dense and small-scale objects.
- Uses Feature Pyramid Network (FPN) on top of ResNet as a backbone.
- Outperforms many one-stage and two-stage detectors.
- Two major improvements:
  - Feature Pyramid Network (FPN):
    - Encoder-decoder architecture with lateral connections.
    - Builds high-level semantic feature maps at all scales.
    - Bottom-up pathway (standard CNN, e.g., ResNet) and top-down pathway.
    - Multiple prediction layers across different scales (e.g., {P2, P3, P4, P5}).
  - Focal Loss:
    - Addresses extreme foreground-background class imbalance.
    - Modulating factor decreases the impact of easy negatives on the loss.
    - Includes a weighting parameter (α) for class frequency adjustment.

### DETR (DEtection TRansformer)
- Combines a backbone for feature extraction with an encoder-decoder architecture based on transformers.
- Generates N proposals based on attention, without spatial anchors or non-maximum suppression.
- Includes a simple feed-forward neural network (ffNN) classifier.
- Simplifies the detection pipeline compared to traditional models like Faster R-CNN.
- Demonstrates superior performance to Faster R-CNN with a ResNet-101 backbone.



# Two Stage detection
First stage: generate dense to sparse region proposals

Second stage: N candidates are further regressed and clasified

These filter out most of negative proposals + better accuracy but slower than one stage
- R-CNN
- Fast R-CNN
- Faster R-CNN
- R-FCN

**R-CNN** (region based convolutional neural networks)
1. Use selective search to identify up to 2000 bounding box object region candidates
2. Use AlexNet-pretrained CNN to extract fatures from selectde + warped regions- generate 4096 dimensional feature vetor for each of 2000 cropped + resize images
3. Classify regions with SVM 
4. Also output  bounding box fixes

Issue: SLOW


**Fast R-CNN**
Core idea:pass image through CNN before cropping
- Input → Convolutional layers → feature map of entire image
- Get Region proposals (like Selective Search)
- RoI Pooling Layer: Convert region proposals to feature vector
	- Divide ROI into fixed number of sections 
	- Do Max pooling over pixels in section
	- Now you have a fixed-szed feature map for each ROI
- SVM can be replaced by FC layer for Classification + bounding box regression
	- Flatten ROI output + pass through 1+ FC

Now dominated by region proposals!

**Faster R-CNN**: enhanced version of Fast R-CNN
- Key difference: instead of Selective search, *region proposal network*
- 4 losses
	- RPN classify objecct / not object
	- RPN regress coordinates
	- Classification score
	- box coordinates

Ignore overalpping proposal with Non-max suprression


# Mask R-CNN
Mask R-CNN is an extension of Faster R-CNN, a popular object detection model, adding the capability for pixel-level segmentation. This makes Mask R-CNN a powerful model for tasks that require identifying not just where objects are (as bounding boxes) but also delineating each object's precise shape (as a mask). :

### Core Idea: 
- Extend Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression.

### Steps in Mask R-CNN:


1. **Input and Feature Extraction**:
   - **Input Image**: The whole image is input to the network.
   - **Convolutional Layers**: The image is passed through a series of convolutional layers, creating a feature map of the entire image.

2. **Region Proposal Network (RPN)**:
   - **Generate Region Proposals**: Unlike Fast R-CNN which uses methods like Selective Search for region proposals, Mask R-CNN uses a Region Proposal Network (RPN). The RPN scans the feature map and outputs a set of rectangular object proposals, each with an objectness score.

3. **RoI Align Layer**:
   - **Upgraded RoI Pooling**: Mask R-CNN introduces RoI Align, an improvement over RoI Pooling used in Fast R-CNN. RoI Align fixes the misalignment issue by using bilinear interpolation to compute the exact values of the input features at four regularly sampled locations in each RoI bin, and then aggregating the results (usually by max pooling).
   - **Fixed-size Feature Maps**: Similar to RoI Pooling, RoI Align ensures a fixed-size feature map for each RoI, which is crucial for the subsequent fully connected layers.

4. **Classification, Bounding Box Regression, and Mask Prediction**:
   - **Fully Connected Layers for Classification and Regression**: Each RoI feature vector is passed through fully connected layers to classify the object and regress the bounding box coordinates.
   - **Parallel Mask Prediction Branch**: In parallel to classification, another branch of the network outputs a binary mask for each RoI. This is done through a Fully Convolutional Network (FCN) that maintains spatial dimensions, unlike the FC layers used for classification and bounding box regression.

5. **Output**:
   - **Class and Bounding Box**: For each region proposal, Mask R-CNN outputs a class label and a bounding box, like Fast R-CNN.
   - **Segmentation Mask**: Additionally, it outputs a binary mask for each object class.

### Advantages of Mask R-CNN:

- **Precision**: RoI Align improves the accuracy of the localization of objects by aligning the extracted features more precisely with the objects.
- **Versatility**: It effectively combines the tasks of object detection (finding bounding boxes) and semantic segmentation (creating a pixel-wise mask), making it a versatile model for a variety of vision tasks.
- **End-to-End Trainable**: Like its predecessors, Mask R-CNN can be trained end-to-end, allowing for joint learning of the various tasks it performs.

In summary, Mask R-CNN takes the image through a convolutional network


**R-FCN**
- In R-CNN family, each detector applies final FC layers to make predictions for each Rol
- **Region based fully convolutoin networks** removes FC layers


**Instance Segmentation**


