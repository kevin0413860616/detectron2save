import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

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
register_coco_instances("peanuttrain", {}, "./coco/train/train.json", "./coco/train")
register_coco_instances("peanutval", {}, "./coco/val/val.json", "./coco/val")

peanut_metadata = MetadataCatalog.get("peanuttrain").set(thing_classes=["leaf","pest","sick"])
dataset_dicts = DatasetCatalog.get("peanuttrain")
import random
import matplotlib.pyplot as plt

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=peanut_metadata, scale=1.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("peanuttrain",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = ("./model/model_mask_r101fpn3x.pkl")
#cfg.MODEL.WEIGHTS = ("./outputmodel/mask_rcnn_50_2000.pth")
cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.001
ITERS_IN_ONE_EPOCH = int(1028 / cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.MAX_ITER =(ITERS_IN_ONE_EPOCH * 500) - 1      # 500 epochs
# 初始學習率
cfg.SOLVER.BASE_LR = 0.001
# 優化器動能
cfg.SOLVER.MOMENTUM = 0.9
# 權重衰減
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
# 學習率衰減倍數
cfg.SOLVER.GAMMA = 0.1
# 迭代到指定次數，學習率進行衰減
cfg.SOLVER.STEPS = (7000,)
# 在訓練前，會進行一個熱身，學習率慢慢增加初始學習率
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# 熱身迭代次數
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1 
# 迭代到指定次數，進行一次評估
cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
#cfg.TEST.EVAL_PERIOD = 100

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)
#cfg.MODEL.RETINANET.NUM_CLASSES = 3 #Retinanet
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #測試的閾值
cfg.DATASETS.TEST = ("coco3/val", )
predictor = DefaultPredictor(cfg)
coco_metadata = MetadataCatalog.get("peanutval").set(thing_classes=["leaf","pest","sick"])

from detectron2.utils.visualizer import ColorMode
dataset_dict = DatasetCatalog.get("peanutval")
for d in random.sample(dataset_dict, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                   metadata=coco_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("peanutval",cfg,False,output_dir="./output")
val_loader = build_detection_test_loader(cfg, "peanutval")

inference_on_dataset(trainer.model, val_loader, evaluator)
