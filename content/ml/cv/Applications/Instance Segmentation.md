---
title: Instance Segmentation
---
Hybrid of semantic segmentaiton and object detetion

pixelwise


(SoTa) Mask R-CNN

Input image → NN → use RPN to project onto feature map (RoIAlign) → CNN  →
- Faster RCN style classification + coodrinates
- Semantic segmentation CNN  (ie. mask prediction)

Kind of
- CNN → RPN to get ROI in feature map
- Then, semantic segmentation for each ROI

