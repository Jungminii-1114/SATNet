# Edited SATNet for Mirror Segmentation

## Overview

This repository contains an experimental reproduction and modified implementation of SATNet for mirror segmentation on the MSD dataset. SATNet is a symmetry-aware transformer-based network designed for detecting mirror regions by exploiting appearance symmetry between an input image and its horizontally flipped counterpart.

This implementation is not the official SATNet codebase. It is a debugging-oriented reproduction that modifies several network components to make the tensor contract clearer and to stabilize decoder behavior.

Official SATNet repository: https://github.com/tyhuang0428/SATNet

## Main Modifications

Compared with the initial reproduction, the edited network introduces the following changes:

- Unified tensor format as `NCHW` across SAAM, CFDM, decoder, and prediction heads.
- Removed internal feature flipping from SAAM.
- Explicitly aligned flipped features before passing them into SAAM and CFDM.
- Added residual connections to SAAM outputs.
- Reworked CFDM top-down fusion with a gated fusion option.
- Replaced shared CCL usage with branch-specific CCL blocks.
- Added residual contrast learning in `CCL_fixed`.
- Unified model output as `[p0, p1, p2, p3]` for deep supervision and head-wise evaluation.

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

The Implemented SATNet significanlty

