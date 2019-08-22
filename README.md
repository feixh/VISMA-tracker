# VISMA-tracker

This is a preemptive release of the code for our ECCV 18 paper:

```
@inproceedings{feiS18,
    title = {Visual-Inertial Object Detection and Mapping},
    author = {Fei, X. and Soatto, S.},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2018}
}
```

The data utilities released earlier can be found [here](https://github.com/feixh/VISMA).


The problem we want to address here is object detection and 6 DoF (Degrees-of-Freedom) object pose estimation.

The code provides a fusion framework (written in C++) to fuse likelihood scores from semantic modules (e.g. object detectors) and low-level image cues (e.g. edges/intensity values) to accompalish this.

For object likelihoods, the system relies on external modules such as Faster R-CNN running in TensorFlow or Pytorch. Since lots of popular deep learning models are written in Python, we provide a message-based inter-process communication facility, enabled by ZMQ (ZeroMQ) library.

For low-level image cues, the code contains implementation of various model-based tracking algorithms which leverage edges/intensity values as evidences and use gradient-based optimization/particle filtering as the underlying inference machinery.


## Applications

In the `app` folder under the project root directory, we provide several applications using our library.

- SORBT_XXX: Single-Object Region-Based Tracker for dataset XXX.
- SODFT_XXX: Single-Object Distance-Field based Tracker for dataset XXX.
- linemod, rigidpose are two model-based tracking datasets.
- visma is our own dataset available [here](https://github.com/feixh/VISMA).

## Build

We include some dependencies in the `thirdparty` folder. Other dependencies (listed below) should be availabe as debian packages. 

To build, simply trigger `build.sh` in the project root directory. Missing packages should be easily resolved by looking up and installing proper packages via your favoriate package manager.

## Dependencies

### Numeric
- GMP: Gnu Multi-Precision
- GLM: OpenGL Mathematics
- Eigen: Template linear algebra library

### Utilities
- tbb: Threading Building Blocks for CPU parallelism from Intel 
- glog: logging
- googletest: unit testing
- gflags: command-line options
- jsoncpp: json for configuration

### Graphics and geometry processing
- igl: Mesh loading and processing
- OpenGL: rendering

### Messaging
- ZMQ: Zero Message Queue
- LCM: Lgihtweight Communications and Marshalling

## Launch faster-rcnn for object likelihood

TODO: will release our customized Detectron software which provides the communication functionality.

To run the following likelihood evaluation process before launching the tracker in detectron root directory with `edge` branch:

```
python2 vlslam_module/infer_process.py --cfg configs/12_2017_baselines/fast_rcnn_R-50-FPN_2x.yaml --output-dir /tmp/detectron-visualizations --wts models/faster_rcnn_R-50-FPN_2x.pkl
```
