# Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment, ArXiv

Welcome to the official page of the paper [Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](https://arxiv.org/pdf/2405.05079).

## Open-Source Implementation

The official implementation will be available soon.

## Abstract

Initialization-free bundle adjustment (BA) remains largely uncharted. While Levenberg-Marquardt algorithm is the golden method to solve the BA problem, it generally relies on a good initialization. In contrast, the under-explored Variable Projection algorithm (VarPro) exhibits a wide convergence basin even without initialization. Coupled with object space error formulation, recent works have shown its ability to solve (small-scale) initialization-free bundle adjustment problem. We introduce Power Variable Projection (PoVar), extending a recent inverse expansion method based on power series. Importantly, we link the power series expansion to Riemannian manifold optimization. This projective framework is crucial to solve large-scale bundle adjustment problem with- out initialization. Using the real-world BAL dataset, we experimentally demonstrate that our solver achieves state-of-the-art results in terms of speed and accuracy. In particular, our work is the first, to our knowledge, that addresses the scalability of BA without initialization and opens new venues for initialization-free Structure-from-Motion.
