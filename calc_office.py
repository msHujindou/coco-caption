from __future__ import print_function
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

pylab.rcParams["figure.figsize"] = (10.0, 8.0)

import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, ".3f")

# create coco object and cocoRes object
coco = COCO("test.caption_coco_format.json")
cocoRes = coco.loadRes("office.json")

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes, "corpus")
# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params["image_id"] = cocoRes.getImgIds()
# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()
