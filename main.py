from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import os.path as op

dirpath = '/home/xiaowh/work/qd_data/getty_images_test'
gt_fpath = op.join(dirpath, 'caption_gt.json')
res_name2path = {n: op.join(dirpath, 'res_{}.json'.format(n)) for n in ['prod',
    'vlp_ce', 'vlp_scst']}
#infile = '/home/xiaowh/work/qd_data/getty_images_test/gettyimages_all_auto_and_manual_captions_vlp.csv'
infile = '/home/xiaowh/work/qd_data/getty_images_test/gettyimages_all_auto_and_manual_captions_vlp_fixed_empty.csv'
eval_res_file = infile + '_eval.json'

def csv2res():
    sep = ','
    gt_col = 3
    res_cols = [[1, 'prod'], [6, 'vlp_ce'], [7, 'vlp_scst']]

    gt_anns = []
    gt_imgs = []
    all_res = {k: [] for _, k in res_cols}
    with open(infile, 'r') as fp:
        for idx, line in enumerate(fp):
            if idx == 0: continue
            parts = line.split(sep)
            key = parts[0]
            gt_anns.append({'image_id': idx, 'key': key, 'caption': parts[gt_col],
                    'id': idx})
            gt_imgs.append({'id': idx})
            for res_col, res_name in res_cols:
                cap = parts[res_col].strip()
                if cap == '':
                    continue
                all_res[res_name].append({'image_id': idx, 'key': key,
                        'caption': cap})
    with open(gt_fpath, 'w') as fp:
        json.dump({'annotations': gt_anns, 'images': gt_imgs,
                'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'}, fp)
    for res_name in res_name2path:
        with open(res_name2path[res_name], 'w') as fp:
            json.dump(all_res[res_name], fp)

csv2res()


#demo_ann_path = 'annotations/captions_val2014.json'
#res_path = 'results/captions_val2014_fakecap_results.json'
#demo_ann = json.load(open(demo_ann_path, 'r'))
#import ipdb;ipdb.set_trace(context=15)
ann_path = gt_fpath
coco = COCO(ann_path)
eval_res = {}
for res_name, res_path in res_name2path.items():
    eval_res[res_name] = {}
    cocoRes = coco.loadRes(res_path)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    for metric, score in cocoEval.eval.items():
        eval_res[res_name][metric] = score

with open(eval_res_file, 'w') as fp:
    json.dump(eval_res, fp, sort_keys=True, indent = 2)
