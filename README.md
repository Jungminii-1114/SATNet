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
