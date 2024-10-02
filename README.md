# Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment, ECCV 2024

Welcome to the official page of the paper [Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](https://arxiv.org/pdf/2405.05079) (ECCV 2024).

## Abstract

Most Bundle Adjustment (BA) solvers like the Levenberg- Marquardt algorithm require a good initialization. Instead, initialization-free BA remains a largely uncharted territory. The under-explored Variable Projection algorithm (VarPro) exhibits a wide convergence basin even without initialization. Coupled with object space error formulation, recent works have shown its ability to solve small-scale initialization-free bundle adjustment problem. To make such initialization-free BA approaches scalable, we introduce Power Variable Projection (PoVar), extending a recent inverse expansion method based on power series. Importantly, we link the power series expansion to Riemannian manifold optimization. This projective framework is crucial to solve large-scale bundle adjustment problems without initialization. Using the real-world BAL dataset, we experimentally demonstrate that our solver achieves state-of-the-art results in terms of speed and accuracy. To our knowledge, this work is the first to address the scalability of BA without initialization opening new venues for initialization-free structure-from-motion.


## Open-Source Implementation

### Install Dependencies and Build External Libraries

This implementation is built on [RootBA](https://github.com/NikolausDemmel/rootba). Please follow the same instructions to install the dependencies.

### How to build it?

**Build external libraries**

Use the build script:

```
./scripts/build-external.sh
```


**Build PoVar**

Use the build script:

```
./scripts/build-rootba-povar.sh [BUILD_TYPE]
```

You can optionally pass the cmake `BUILD_TYPE` used to compile RootBA
as the first argument. If you don't pass anything the default is
`Release`. The cmake build folder is `build`, inside the project
root. This build script will use `ccache` and `ninja` automaticaly if
they are found on `PATH`.

### Running PoVar

#### Dataset

You can download the BAL dataset on the [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/) webpage. 
Before using our solver, we randomly initialize a projective camera model, with the option ```--create-dataset```:
```
./bin/bal --input /venice/problem-89-110973-pre.txt --create-dataset
```

The randomized dataset is created in the new folder ```data_custom/``` on the root.

#### Solving initialization-free stratified BA

In line with [our paper](https://arxiv.org/abs/2405.05079), the stratified BA includes two steps.

(a) We propose four different solvers ```--solver-type-step-1``` for the nonlinear separable optimization problem:
* ```POWER_VARPROJ```: Variable projection with power series expansion (by default)
* ```POWER_BUNDLE_ADJUSTMENT```: Levenberg-Marquardt with power series expansion (see PoBA https://arxiv.org/abs/2204.12834)
* ```PCG```: Variable projection with preconditioned conjugate gradients
* ```CHOLESKY```: Variable projection with Cholesky factorization


(b) We propose two different solvers ```--solver-type-step-2``` for the projective refinement problem:

* ```RIPOBA```: Riemannian manifold framework for Levenberg-Marquardt with power series (by default)
* ```RIPCG```: Riemannian manifold framework for Levenberg-Marquardt with preconditioned conjugate gradients


The implementation uses ```double``` precision.

#### Command line
Once the random initialization has been done, you can run the two solvers in a row with, for instance:
```
./bin/bal --num-threads 4 --input /data_custom/problem-89-110973-pre.txt --solver-type-step-1 POWER_SCHUR_COMPLEMENT --solver-type-step-2 RIPOBA 
```

The command ```--help``` will provide some explanations about the different options.
In particular, you can set, among others:
* ```--max-num-iterations-step-1```, ```--max-num-iterations-step-2```: maximum number of outer iterations for each step (by default, 50).
* ```--power-sc-iterations```: maximum order of power series (by default, 20).
* ```--residual-robust-norm```: NONE (by default), CAUCHY, HUBER.
* ```--alpha```: weight of the affine part in pOSE formulation (by default, 0.1).

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{weber2024power,
  title={Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment},
  author={Weber, Simon and Hong, Je Hyeong and Cremers, Daniel},
  journal={arXiv preprint arXiv:2405.05079},
  year={2024}
}

@inproceedings{weber2023power,
  title={Power bundle adjustment for large-scale 3d reconstruction},
  author={Weber, Simon and Demmel, Nikolaus and Chan, Tin Chon and Cremers, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={281--289},
  year={2023}
}
```

## License

The code of the RootBA project is licensed under a [BSD 3-Clause
License](LICENSE).

Parts of the code are derived from [Ceres
Solver](https://github.com/ceres-solver/ceres-solver). Please also
consider licenses of used third-party libraries. See
[ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
