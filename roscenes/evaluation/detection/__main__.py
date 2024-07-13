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


# if __name__ == "__main__":
#     import sys
#     import pickle
#     import json
#     from time import time

#     CLASS_ID_MAP = {n: i for i, n in enumerate(["other", "truck", "bus", "van", "car"])}

#     # with open("gt.json", "r") as fp:
#     #     gt = json.load(fp)["GT"]
#     # with open("pred.json", "r") as fp:
#     #     pred = json.load(fp)["results"]

#     # views = list()
#     # for i, (sampleToken, gts) in enumerate(gt.items()):
#     #     boxes = list()
#     #     velocities = list()
#     #     labels = list()
#     #     for g in gts:
#     #         xyz = g["translation"]
#     #         wlh = g["size"]
#     #         q = g["rotation"]
#     #         vel = g["velocity"]
#     #         label = CLASS_ID_MAP[g["detection_name"]]
#     #         boxes.append(xyz + wlh + q)
#     #         velocities.append(vel)
#     #         labels.append(label)
#     #     boxes = np.float64(boxes)
#     #     velocities = np.float64(velocities)
#     #     labels = np.int64(labels)
#     #     view = ViewsPrediction(i, boxes, labels, np.ones_like(labels), velocities, sampleToken)
#     #     views.append(view)
#     # clip = ClipPrediction(views, "sample")
#     # scene = ScenePrediction([clip])

#     # viewsPredictions = list()
#     # for i, sampleToken in enumerate(gt):
#     #     boxes = list()
#     #     velocities = list()
#     #     labels = list()
#     #     scores = list()
#     #     preds = pred[sampleToken]
#     #     for p in preds:
#     #         xyz = p["translation"]
#     #         wlh = p["size"]
#     #         q = p["rotation"]
#     #         vel = p["velocity"]
#     #         label = CLASS_ID_MAP[p["detection_name"]]
#     #         boxes.append(xyz + wlh + q)
#     #         velocities.append(vel)
#     #         labels.append(label)
#     #         scores.append(p["detection_score"])
#     #     boxes = np.float64(boxes)
#     #     velocities = np.float64(velocities)
#     #     labels = np.int64(labels)
#     #     scores = np.float64(scores)
#     #     view = ViewsPrediction(i, boxes, labels, scores, velocities, sampleToken)
#     #     viewsPredictions.append(view)
#     # predClip = ClipPrediction(viewsPredictions, "sample")
#     # predScene = ScenePrediction([predClip])

#     start = time()

#     scene = Scene.load(sys.argv[1])

#     # generate fake data for prediction
#     fakeScene = list()
#     for clip in scene.clips:
#         fakeClip = list()
#         for views in clip:
#             fake = ViewsPrediction(views.timeStamp, views.boxes3D.copy(), views.labels.copy(), np.random.rand(len(views.boxes3D)), views.velocities.copy(), views.token)
#             fakeClip.append(fake)
#         fakeClip = ClipPrediction(fakeClip, clip.token)
#         fakeScene.append(fakeClip)
#     predScene = ScenePrediction(fakeScene)

#     # with open(sys.argv[2], "rb") as fp:
#     #     prediction = Converter()(pickle.load(fp), scene)
#     evaluator = MultiView3DEvaluator(MultiView3DEvaluationConfig(
#         ["other", "truck", "bus", "van", "car"],
#         [0.5, 1., 2., 4.],
#         2.,
#         ThresholdMetric.CenterDistance,
#         1000000,
#         0.0,
#         # [-380., -25., -3., 380., 25., 3.],
#         [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9],
#         []
#     ))
#     result = evaluator(scene, predScene)
#     print(result)
#     print(result.summary)
#     end = time()
#     print("time elapsed:", end - start)