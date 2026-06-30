# Edited SATNet for Mirror Segmentation

## Overview

This repository contains an experimental reproduction and modified implementation of SATNet for mirror segmentation on the MSD dataset. SATNet is a symmetry-aware transformer-based network designed for detecting mirror regions by exploiting appearance symmetry between an input image and its horizontally flipped counterpart.

This implementation is not the official SATNet codebase. It is a debugging-oriented reproduction that modifies several network components to make the tensor contract clearer and to stabilize decoder behavior.

Official SATNet repository: https://github.com/tyhuang0428/SATNet

## Main Components

### Swin Transforer Backbone

The backbone extracts multil-scale visual features from both the original image and its horizontally flipped counterpart. In this implementation, Swin-S is used as the feature extractor.
The backbone outputs four feature levels, which are later used by SAAM and CFDM.

### SAAM: Symmetry-Aware Attention Module

SAAM is designed to exploit the symmetry relationship between the original image and its flipped image. 
It receives feature maps from both branches after alignment and applies attention-based feature ineraction. This helps the network capture mirror-related symmetry cues that may not be obvious from the original iamge alone.

In this implementation, flippped features are explicitly aligned before entering SAAM, and residual connections are used to preserve the original backbone features.

### CFDM: Contrast and Fusion Decoder Module

CFDM progressively fuses high-level semantic features with lower-level spatial features. It receives features from the original and flipped branches, then combines them with the previous decoder output in a top-down manner.

This module is respoinsible for producing multi-scale decoder outputs. In this implementation, a gated top-down fusion strategy is used to stabliize feature propagation across decoder stages.

### CCL: Contrast Context Learning Block

CCL is used inside CFDM to enhance contrast-aware contextual representations. It compares local and contextual responses to emphasize discriminative mirror-related regions.

In this implementation, residual contrast learning is used so that the original feature information is preserved while contrast-enhanced responses are added.


### Prediction Heads

The decoder produces four prediction heads: `p0`, `p1`, `p2`, and `p3`.

In this implementation, `p3` is the coarsest prediction from the deepest decoder stage, while `p0` is the final high-resolution prediction after all top-down CFDM stages. Therefore, `p0` is used as the final output for test evaluation.

---- 

The edited model is implemented in:

```text
sample_network_edited.py
```

The main training model is
```python
TrainableSATNetWithSwinEdited
```
## Experimental Setup
| Item | Setting |
| :--- | :--- |
| Dataset | MSD |
| Train / Validation / Test | 2757 / 306 / 955 |
| Input size | 512x512 |
| Backbone | Swin-S |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Optimizer updates | 20,000 |
| Evaluation metrics | IoU, F-measure, MAE |

## Evaluation Metrics
The model is evaluated using the same major metrics reported in STANet:
| Metric | Description |
| :--- | :--- |
| IoU | Intersectino over Union between predicted mask and ground-truth mask |
| F-measure | Precision-recall based segmentation quality metric |
| MAE | Mean Absolute Error between prediction probability map and ground-truth mask |

Higher IoU and F-measure indicate better performance, while lower MAE indicates better performance.

## Results on MSD Test set
| Method | Dataset | IoU | F-measure | MAE |
|:--- | :--- | :--- | :--- | :--- |
| Official SATNet | MSD | 85.41 | 0.9222 | 0.033 |
| Implemented SATNet | MSD | **63.54** | **0.8346** | **0.0852** |

The official SATNet result is reported in the SATNet Github Repository.
<br>Source: https://github.com/tyhuang0428/SATNet

## Head-wse Evaluation
| Head | IoU | F-measure | MAE | Best Threshold |
| :--- | :--- | :--- | :--- | :--- |
| p0 | 63.54 | 0.8346 | 0.0852 | 0.6235 |
| p1 | 63.37 | 0.8331 | 0.0855 | 0.6118 |
| p2 | 62.75 | 0.8342 | 0.0879 | 0.6588 |
| p3 | 63.01 | 0.8378 | 0.0874 | 0.6392 |

Among the four prediction heads, `p0` is used as the final prediction head because it achieves the best IoU and MAE. 

## Discussion

The edited SATNet achieved meaningful mirror segmentation performance on the MSD test set, with `IoU 63.54`, `F-measure 0.8346`, and `MAE 0.0852`. The four prediction heads produced similar performance, and the final high-resolution head `p0` achieved the best IoU and MAE among them.

This result suggests that the decoder produces consistent multi-scale predictions and that the final top-down prediction head is functioning as intended. The head-wise evaluation also shows that the model does not rely on a single intermediate head, but instead maintains relatively stable performance across decoder stages.

However, the current result is still below the official SATNet performance reported on MSD. The remaining gap may be caused by differences in the exact decoder implementation, contrast learning design, data augmentation pipeline, learning schedule, backbone implementation, and pretrained weight loading strategy.

Future work should focus on further aligning this implementation with the official SATNet training and architectural details, especially the CFDM and CCL designs, backbone initialization, and data processing pipeline.
