#    RoScenes
#    Copyright (C) 2024  Alibaba Cloud
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import json
import math
import os
import shutil
import cv2
import numpy as np
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets import DATASETS
from shapely.geometry import Polygon

from roscenes.data import Clip, Frame, Scene, ConcatScene
from roscenes.data.metadata import Split
from roscenes.evaluation.detection import MultiView3DEvaluator, ThresholdMetric, DetectionEvaluationConfig
from roscenes.evaluation.detection import Prediction
from roscenes.transform import xyzwlhq2kitti, kitti2xyzwlhq

COLOR_PALETTE = [
    [0, 0, 0],
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 255, 255]
]


@DATASETS.register_module(force=True)
class RoScenesDataset(Custom3DDataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    .. code-block:: none

    [
        {'sample_idx':
         'lidar_points': {'lidar_path': velodyne_path,
                           ....
                         },
         'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
                   'gt_bboxes_3d':  <np.ndarray> (n, 7)
                   'gt_names':  [list]
                   ....
               }
         'calib': { .....}
         'images': { .....}
        }
    ]

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = [
        "other",
        "truck",
        "bus",
        "van"
        "car",
    ]

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    data_infos: Scene

    def __init__(self,
                 data_root,
                 ann_file,
                 data_list=None,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 use_valid_flag=False,
                 test_mode=False):
        super().__init__(data_root, ann_file, pipeline, classes, modality, box_type_3d, filter_empty_gt, test_mode)
        self.seq_split_num = 1
        self._set_sequence_group_flag()

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx].previous is not None:
                # Not first frame and previous is None  -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            Scene
        """
        scene = Scene.load(self.data_root)
        print('load a scene with length:', len(scene))
        return scene

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        # a frame
        frame: Frame = self.data_infos[index]

        input_dict = dict(
            sample_idx=index,
            scene_token=frame.parent.token,
            timestamp=frame.timeStamp / 1e6)

        clip: Clip = frame.parent

        # NOTE: Copy it to avoid inplace manipulation on raw data --- This causes a messd up.
        intrinsics = [c.intrinsic.copy() for c in clip.cameras.values()]
        extrinsics = [c.extrinsic.copy() for c in clip.cameras.values()]
        world2image = [c.world2image.copy() for c in clip.cameras.values()]
        input_dict.update(dict(
            img_timestamp=[frame.timeStamp / 1e6 for _ in range(len(frame.imagePaths))],
            img_filename=list(frame.images.values()),
            lidar2img=world2image,
            lidar2cam=extrinsics,
            cam_intrinsic=intrinsics
        ))

        if not self.test_mode:
            gt_bboxes = LiDARInstance3DBoxes(np.concatenate([xyzwlhq2kitti(frame.boxes3D), frame.velocities], -1).astype(np.float32), box_dim=7 + 2, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            annos = dict(
                gt_bboxes_3d=gt_bboxes,
                gt_labels_3d=frame.labels.copy(),
                gt_names=self.CLASSES,
                bboxes_ignore=None
            )
            input_dict['ann_info'] = annos
        return input_dict

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir="results",
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(self.data_infos, ConcatScene):
            metadata = self.data_infos.scenes[0].metadata
        else:
            metadata = self.data_infos.metadata

        visFolder = out_dir
        os.makedirs(visFolder, exist_ok=True)
        previousClip = None
        predictionList = list()
        clips = list()

        for i, res in enumerate(results):
            frame = self.data_infos[i]
            boxes_3d = res['pts_bbox']['boxes_3d']
            scores_3d = res['pts_bbox']['scores_3d']
            labels_3d = res['pts_bbox']['labels_3d']
            # [N, 7+2]
            xyzwlhr, velocities = boxes_3d.tensor[:, :7].detach().clone(), boxes_3d.tensor[:, 7:9].detach().clone()

            xyzwlhq = kitti2xyzwlhq(xyzwlhr.cpu().numpy().copy())

            # if i % 60 == 58:
            #     boxes2vis = boxes_3d[scores_3d > 0.3]
            #     scores2vis = scores_3d[scores_3d > 0.3]
            #     labels2vis = labels_3d[scores_3d > 0.3]
            #     # projectedResults = view.parent.projection(kitti2corners(boxes_3d.tensor.detach().clone().cpu().numpy()[..., :7]))
            #     projectedResults = view.parent.projection(boxes2vis.corners.detach().cpu().numpy())
            #     for k, (imagePath, (boxes, vis)) in enumerate(zip(view.images.values(), projectedResults)):
            #         img = cv2.imread(imagePath)
            #         cleanImg = img.copy()

            #         sortIds = np.argsort(-np.mean(boxes[..., -1], -1))
            #         # [N, 8, 2]
            #         boxes = boxes[sortIds, ..., :2]
            #         scores2vis = scores2vis[sortIds]
            #         labels2vis = labels2vis[sortIds]
            #         vis = vis[sortIds]

            #         # [4] in xy format
            #         for box3d, score, label in zip(boxes[vis], scores2vis[vis], labels2vis[vis]):
            #             # crop the clean object region
            #             # paste to current image
            #             # then draw line
            #             objectPoly = Polygon(box3d)
            #             objectPoly = np.array(objectPoly.convex_hull.exterior.coords, dtype=np.int32)
            #             mask = np.zeros_like(cleanImg[..., 0])
            #             cv2.drawContours(mask, [objectPoly], -1, (255, 255, 255), -1, cv2.LINE_AA)
            #             # print(img.shape, cleanImg.shape, mask.shape)
            #             fg = cv2.bitwise_and(cleanImg, cleanImg, mask=mask)
            #             bg = (img * (1 - mask[..., None] / 255.)).astype(np.uint8)
            #             img = fg + bg
            #             cv2.polylines(img, [box3d[:4].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
            #             cv2.polylines(img, [box3d[4:].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
            #             cv2.polylines(img, [box3d[[0, 1, 5, 4]].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
            #             cv2.polylines(img, [box3d[[2, 3, 7, 6]].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)

            #             cv2.putText(img, f"{score:.2f}", box3d[4, :2].astype(np.int32), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
            #             cv2.putText(img, f"{score:.2f}", box3d[4, :2].astype(np.int32), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), 2, cv2.LINE_AA)

            #         os.makedirs(os.path.join(visFolder, str(i)), exist_ok=True)
            #         cv2.imwrite(os.path.join(visFolder, str(i), f"{view.token}_{k}.jpg"), img)


            prediction = Prediction(
                            timeStamp=frame.timeStamp,
                            boxes3D=xyzwlhq,
                            velocities=velocities.cpu().numpy().copy(),
                            labels=labels_3d.cpu().numpy().copy(),
                            scores=scores_3d.cpu().numpy().copy(),
                            token=frame.token
                        )
            predictionList.append(prediction)

        groundtruth = self.data_infos

        evaluator = MultiView3DEvaluator(DetectionEvaluationConfig(
            self.CLASSES,
            [0.5, 1., 2., 4.],
            2.,
            ThresholdMetric.CenterDistance,
            500,
            0.0,
            [-400., -40., 0., 400., 40., 6.],
            ["ATE", "ASE", "AOE"]
        ))
        result = evaluator(groundtruth, predictionList)

        summary = result.summary

        with open(os.path.join(out_dir, "result.json"), "w") as fp:
            json.dump(summary, fp)
        print(result)
        return summary