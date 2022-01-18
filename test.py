# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

register_coco_instances("peanutval", {}, "./newcoco/val/val.json", "./newcoco/val")

import random
import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

coco_metadata = MetadataCatalog.get("peanutval").set(thing_classes=["leaf","pest","sick"],thing_colors=[(0,0,255),(0,255,0),(0,255,255)])
#coco_metadata = MetadataCatalog.get("peanutval").set(thing_classes=["leaf",0,1])

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join("./outputmodel/mask_rcnn_101_3000.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7#测试的阈值
cfg.MODEL.ROI_HEADS.NUM_CLASSES=3
cfg.DATASETS.TEST = ("newcoco/val", )
predictor = DefaultPredictor(cfg)
#随机选部分图片，可视化
from detectron2.utils.visualizer import ColorMode
dataset_dict = DatasetCatalog.get("peanutval")
for d in random.sample(dataset_dict, 40):
   im = cv2.imread(d["file_name"])
   b,g,r = cv2.split(im)
   img_rgb = cv2.merge([r,g,b])
# print(im)
  # print(d["file_name"][15:])
   outputs = predictor(img_rgb)
   v = Visualizer(img_rgb[:, :, ::-1],
                   metadata=coco_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.SEGMENTATION # 去除非气球区域的像素颜色.
    )
   v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   #v = v.draw_instance_predictions(outputs["instances"][outputs["instances"].pred_classes == 2].to("cpu"))
   plt.imshow(v.get_image()[:, :, ::-1])
   plt.savefig("out/mask_rcnn_101/3000/all/"+"3000_"+"all"+d["file_name"][17:])
  # plt.show()

