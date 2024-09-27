# Examples
This folder provides typical use cases in companion with the elegant `mmdet3d` package.

## Preparation
The example is based on the `mmdet3d 1.0.0rc6`, which is compatible with most popular BEV perception algorithms, like DETR3D, PETR, StreamPETR, BEVFormer, etc.

Please install mmdet3d following the official instructions.

Note that higher version (`mmdet3d>1.0.0`) is not tested, the given example may fail to run.

## Integrate into mmdet3d dataset.
The file [`examples/mmdet3d/mmdet3d_plugin/datasets/roscenes_dataset.py`](mmdet3d/mmdet3d_plugin/datasets/roscenes_dataset.py) shows how to implement a mmdet3d-compatible dataset using `Custom3DDataset`.

To use it, you need to declare with key in the training or evaluation configs. An example config is provided as [`examples/mmdet3d/configs/detr3d_roscene_res101.py`](mmdet3d/configs/detr3d_roscene_res101.py), which should be compatible with DETR3D model.

## Evaluation
An evaluation code is delivered together with `roscenes_dataset.py`. You can find it in `RoSceneDataset.evaluate()` function.

A detailed guide for performing evaluation is placed under [`roscenes/evaluation/README.md`](/roscenes/evaluation/README.md).