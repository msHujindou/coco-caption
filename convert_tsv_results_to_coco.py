import pandas as pd
import json


def extract_tsv_prediction_results_to_pdf(file_name):
    pdf = pd.read_csv(file_name, header=None, names=["img_key", "pred"], delimiter="\t")
    extracted = pdf.apply(lambda x: json.loads(x["pred"]), axis=1, result_type="expand")
    pdf["results"] = extracted.apply(lambda x: x["description"]["captions"], axis=1)
    return pdf, extracted


def export_pdf_to_json(pdf):
    resjson = []
    for idx, row in pdf.iterrows():
        for itm in row["results"]:
            resjson.append({"image_id": row["img_key"], "caption": itm["text"]})
        pass
    with open("getty.json", "w") as fw:
        json.dump(resjson, fw)
        pass
    return resjson


df, _ = extract_tsv_prediction_results_to_pdf(
    "getty_images_clean.v3.2-nocl.20210507.tsv"
)
js = export_pdf_to_json(df)
print(js)
