This is the code for our ECCV 18 paper:

```
@inproceedings{feiS18,
    title = {Visual-Inertial Object Detection and Mapping},
    author = {Fei, X. and Soatto, S.},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2018}
}
```

I'll clean up the code base soon. The data utilities released earlier can be found [here](https://github.com/feixh/VISMA).


## Applications

- SOT: Single Object Tracking
- RT: Region-based Tracker
- DFT: Distance Field based Tracker
- linemod and rigidpose are two datasets on model-based tracking


## Dependencies

- GMP: Gnu Multi-Precision
- GLM: OpenGL Mathematics
- Folly: Facebook utilities



## Launch faster-rcnn for object likelihood
To run the following likelihood evaluation process before launching the tracker in detectron root directory with `edge` branch:

```
python2 vlslam_module/infer_process.py --cfg configs/12_2017_baselines/fast_rcnn_R-50-FPN_2x.yaml --output-dir /tmp/detectron-visualizations --wts models/faster_rcnn_R-50-FPN_2x.pkl
```
