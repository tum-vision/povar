# Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment, ECCV 2024

Welcome to the official page of the paper [Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](https://arxiv.org/pdf/2405.05079).

## Open-Source Implementation

The official implementation will be available soon.

## Abstract

Most Bundle Adjustment (BA) solvers like the Levenberg- Marquardt algorithm require a good initialization. Instead, initialization-free BA remains a largely uncharted territory. The under-explored Variable Projection algorithm (VarPro) exhibits a wide convergence basin even without initialization. Coupled with object space error formulation, recent works have shown its ability to solve small-scale initialization-free bundle adjustment problem. To make such initialization-free BA approaches scalable, we introduce Power Variable Projection (PoVar), extending a recent inverse expansion method based on power series. Importantly, we link the power series expansion to Riemannian manifold optimization. This projective framework is crucial to solve large-scale bundle adjustment problems without initialization. Using the real-world BAL dataset, we experimentally demonstrate that our solver achieves state-of-the-art results in terms of speed and accuracy. To our knowledge, this work is the first to address the scalability of BA without initialization opening new venues for initialization-free structure-from-motion.

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{weber2024power,
  title={Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment},
  author={Weber, Simon and Hong, Je Hyeong and Cremers, Daniel},
  journal={arXiv preprint arXiv:2405.05079},
  year={2024}
}

@inproceedings{weber2023poba,
 author = {Simon Weber and Nikolaus Demmel and Tin Chon Chan and Daniel Cremers},
 title = {Power Bundle Adjustment for Large-Scale 3D Reconstruction},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2023}
}
```
