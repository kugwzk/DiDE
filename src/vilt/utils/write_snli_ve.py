import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from collections import defaultdict


def process(root, iden, row):
    texts = [r["sentence"] for r in row]
    labels = [r["label"] for r in row]

    path = f"{root}/images/{iden}"
    with open(f"{path}.jpg", "rb") as fp:
        img = fp.read()

    return [img, texts, labels, iden]


def make_arrow(root, dataset_root):
    with open(f"{root}/train.json", "r") as fr:
        train_data = json.load(fr)
    with open(f"{root}/dev.json", "r") as fr:
        dev_data = json.load(fr)
    with open(f"{root}/test.json", "r") as fr:
        test_data = json.load(fr)

    splits = [
        "train",
        "dev",
        "test",
    ]

    datas = [
        train_data,
        dev_data,
        test_data,
    ]

    annotations = dict()

    for split, data in zip(splits, datas):
        _annot = defaultdict(list)
        for row in tqdm(data):
            _annot[row["image"]].append(row)
        annotations[split] = _annot

    for split in splits:
        bs = [
            process(root, iden, row) for iden, row in tqdm(annotations[split].items())
        ]

        dataframe = pd.DataFrame(
            bs, columns=["image", "questions", "answers", "identifier"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/snli_ve_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
