# Evaluation Suite
We provide RoScenes evaluation suite, which is compatible with NDS and should produce identical results with same input and same config.

Why we decide to implement a new one other than using the originals? We found it is too slow when using nuScenes' kit if there is too many boxes in the clip (It is RoScenes' nature!). Therefore, we create a new one from scratch, using multiple optimization tricks (vectorized computation,  joblib parallel, etc.) for speedup.

***Finally we reached about 15x faster evaluation.***

If you want to migrate from the original NDS eval to our new eval, please follow the guide.

## Step-by-Step Implementation

1. Firstly, instantiate the evaluator with config.
```python
from roscenes.evaluation.detection import MultiView3DEvaluator, ThresholdMetric, DetectionEvaluationConfig
from roscenes.evaluation.detection import Prediction

CLASSES = [
    #### PUT YOUR CLASSES HERE ####
]

# The config is brought from the official NDS benchmark
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json
evaluator = MultiView3DEvaluator(DetectionEvaluationConfig(
    CLASSES,
    # box matching thresholds
    [0.5, 1., 2., 4.],
    # true positive threshold
    2.,
    ThresholdMetric.CenterDistance,
    # max boxes per frame
    500,
    # min score filter
    0.0,
    # detection range
    [-400., -40., 0., 400., 40., 6.],
    # metrics, support ATE, ASE, AOE, AVE
    ["ATE", "ASE", "AOE"]
))
```

2. Prepare groundtruth
```python
# groundtruth could be a `RoScene.data.Scene` object.
# Also, if you want to use your own data, you only need to fill the data into a list of predictions as groundtruth.

# Here, we use prediction object for filling groundtruth data. It is ok.
from roscenes.evaluation.detection import Prediction
from roscenes.transform import xyzwlhq2kitti, kitti2xyzwlhq

gt_data = list()
for gt in groundtruths:
    gt_data.append(Prediction(
                        timeStamp=gt['timestamp'],
                        # convert kitti's xyzwlhr format to xyzwlhq, q for quaternion.
                        # [N, 10] array
                        boxes3D=kitti2xyzwlhq(gt['kitti']),
                        # [N, 2] array for vx, vy
                        velocities=gt['velocities'],
                        # [N] int array
                        labels=gt['labels'],
                        # not used
                        scores=np.ones_like(gt['labels']),
                        # unique frame ID
                        token=gt['token']
                    ))
```

3. Similarly, the predictions is constructed:
```python
predictions = list()
for pr in preds:
    predictions.append(Prediction(
                        timeStamp=pr['timestamp'],
                        # convert kitti's xyzwlhr format to xyzwlhq, q for quaternion.
                        # [N, 10] array
                        boxes3D=kitti2xyzwlhq(pr['kitti']),
                        # [N, 2] array for vx, vy
                        velocities=pr['velocities'],
                        # [N] int array
                        labels=pr['labels'],
                        # [N] 0~1 array
                        scores=pr['scores'],
                        # unique frame ID, aligned with GT
                        token=gt['token']
                    ))
```

4. Then, the only thing to do is to call evaluator.
```python
# run evaluation
result = evaluator(gt_data, predictions)
# OR, the gt_data could be a Scene object, remember to check: tokens in predictions are aligned with GT.
result = evaluator(Scene.load('xxxxx/test/*'), predictions)


# a dict for saving result
summary = result.summary
### NOTE: you can also pretty-print the result in a table
print(result)
```
The result is printed as:

```txt
+-------+--------+--------+--------+--------+--------+
| Class |  NDS   |  mAP   |  mATE  |  mASE  |  mAOE  |
+=======+========+========+========+========+========+
|  All  | 0.3953 | 0.1088 | 0.7406 | 0.1450 | 0.0688 |
+-------+--------+--------+--------+--------+--------+
+-------+--------+-----------------------------------+--------+--------+--------+
|       |        |                AP                 |        |        |        |
| Class |  NDS   +--------+--------+--------+--------+  ATE   |  ASE   |  AOE   |
|       |        |  0.5m  |  1.0m  |  2.0m  |  4.0m  |        |        |        |
+=======+========+--------+--------+--------+--------+========+========+========+
| other |                                Ignored                                |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+
| truck | 0.4149 | 0.0042 | 0.0624 | 0.1917 | 0.3600 | 0.7760 | 0.1505 | 0.0478 |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+
|  bus  | 0.3630 | 0.0000 | 0.0177 | 0.0651 | 0.1357 | 0.7634 | 0.1495 | 0.0733 |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+
|  van  | 0.3738 | 0.0005 | 0.0238 | 0.0742 | 0.1637 | 0.7008 | 0.1762 | 0.0766 |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+
|  car  | 0.4296 | 0.0081 | 0.0604 | 0.1702 | 0.4024 | 0.7220 | 0.1039 | 0.0776 |
+-------+--------+--------+--------+--------+--------+--------+--------+--------+
```