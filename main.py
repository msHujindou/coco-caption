from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

pylab.rcParams["figure.figsize"] = (10.0, 8.0)

import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, ".3f")
import os.path as op


def get_url_lst():
    url_file = "/home/xiaowh/work/qd_data/GettyImages/url.label.tsv"
    url_lst = "/home/xiaowh/work/qd_data/GettyImages/gettyimages.url.lst"
    urls = []
    with open(url_file, "r") as fp:
        for line in fp:
            parts = line.split("\t")
            urls.append(parts[0])
    with open(url_lst, "w") as fp:
        for l in urls:
            fp.write(l)
            fp.write("\n")


def get_image_key_to_id(gt_file):
    gt = json.load(open(gt_file))
    gt_anns = gt["annotations"]
    image_key2id = {}
    for ann in gt_anns:
        image_key2id[ann["image_id"]] = ann["image_id"]
    return image_key2id


def convert_tsv_to_coco_format(
    res_tsv, gt_file, outfile, sep="\t", key_col=0, cap_col=1
):
    image_key2id = get_image_key_to_id(gt_file)
    lower_key2key = {k.lower(): k for k in image_key2id}
    results = []
    http_prefix = (
        "https://osizewuspersimmon001.blob.core.windows.net/m365content/publish/"
    )
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            cap = json.loads(parts[cap_col])
            # cap = parts[cap_col]
            if key.startswith(http_prefix):
                key = key[len(http_prefix) :]
                key = lower_key2key[key]

            results.append(
                {
                    "image_id": image_key2id[key],
                    "key": key,
                    # "caption": json.dumps(cap["description"]["captions"]),
                    "caption": cap["description"]["captions"][0]["text"],
                    # "caption": cap,
                }
            )
    with open(outfile, "w") as fp:
        json.dump(results, fp)


def csv2res():
    dirpath = "/home/xiaowh/work/qd_data/getty_images_test"
    # infile = '/home/xiaowh/work/qd_data/getty_images_test/gettyimages_all_auto_and_manual_captions_vlp.csv'
    infile = "/home/xiaowh/work/qd_data/getty_images_test/gettyimages_all_auto_and_manual_captions_vlp_fixed_empty.csv"

    sep = ","
    gt_col = 3
    res_cols = [[1, "prod"], [6, "vlp_ce"], [7, "vlp_scst"]]

    gt_anns = []
    gt_imgs = []
    all_res = {k: [] for _, k in res_cols}
    with open(infile, "r") as fp:
        for idx, line in enumerate(fp):
            if idx == 0:
                continue
            parts = line.split(sep)
            key = parts[0]
            gt_anns.append(
                {"image_id": idx, "key": key, "caption": parts[gt_col], "id": idx}
            )
            gt_imgs.append({"id": idx})
            for res_col, res_name in res_cols:
                cap = parts[res_col].strip()
                if cap == "":
                    continue
                all_res[res_name].append({"image_id": idx, "key": key, "caption": cap})
    with open(gt_fpath, "w") as fp:
        json.dump(
            {
                "annotations": gt_anns,
                "images": gt_imgs,
                "type": "captions",
                "info": "dummy",
                "licenses": "dummy",
            },
            fp,
        )
    for res_name in res_name2path:
        with open(res_name2path[res_name], "w") as fp:
            json.dump(all_res[res_name], fp)


def main(gt_fpath, res_name2path, eval_res_file):
    coco = COCO(gt_fpath)
    eval_res = {}
    for res_name, res_path in res_name2path.items():
        eval_res[res_name] = {}
        cocoRes = coco.loadRes(res_path)

        # create cocoEval object by taking coco and cocoRes
        cocoEval = COCOEvalCap(coco, cocoRes, "corpus")

        # evaluate on a subset of images by setting
        # cocoEval.params['image_id'] = cocoRes.getImgIds()
        # please remove this line when evaluating the full validation set
        cocoEval.params["image_id"] = cocoRes.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        cocoEval.evaluate()
        for metric, score in cocoEval.eval.items():
            eval_res[res_name][metric] = score

    with open(eval_res_file, "w") as fp:
        json.dump(eval_res, fp, sort_keys=True, indent=2)


if __name__ == "__main__":
    # get_url_lst()
    dirpath = "./"
    gt_fpath = op.join(dirpath, "getty.test.caption_coco_format.json")
    # for user inserted clean
    # gt_fpath = op.join(dirpath, "test.caption_coco_format.json")
    eval_res_file = op.join(dirpath, "eval_res.json")

    new_model = "prod"

    res_name2path = {n: op.join(dirpath, "res_{}.json".format(n)) for n in ["prod"]}

    new_res_file = op.join(dirpath, "getty_images_clean.v3.2-nocl.20210507.tsv")
    # for user inserted clean
    # new_res_file = op.join(dirpath, "user_inserted_clean.v3.2-nocl.20210507.tsv")
    new_outfile = op.join(dirpath, "res_{}.json".format(new_model))
    convert_tsv_to_coco_format(new_res_file, gt_fpath, new_outfile)

    main(gt_fpath, res_name2path, eval_res_file)
