import re

import numpy as np
import pandas as pd


def extract_media(text):
    tmp_list = sum([re.findall(r"(.*)によると", sent) for sent in text.split("。")], [])
    return sum([tl.replace("など", "").split("、") for tl in tmp_list], [])


if __name__ == "__main__":
    test = pd.read_csv("../input/nishika-fakenews/test.csv")

    predfull = np.load("test_prediction_full0.npy")

    pred0 = np.load("test_prediction_fold0.npy")
    pred1 = np.load("test_prediction_fold1.npy")
    pred2 = np.load("test_prediction_fold2.npy")
    pred3 = np.load("test_prediction_fold3.npy")
    pred4 = np.load("test_prediction_fold4.npy")

    results = (pred0 + pred1 + pred2 + pred3 + pred4) / 5
    results = predfull * 0.3 + results * 0.7

    test["isFake"] = np.argmax(results, axis=-1)

    # post processing from oof analysis
    test["media"] = [extract_media(text) for text in test["text"]]
    test.loc[test["media"].map(lambda x: "C" in x), "isFake"] = 1
    test.loc[test["media"].map(lambda x: "47" in x), "isFake"] = 1
    test.loc[test["text"].str.contains("「」"), "isFake"] = 1
    test.loc[test["text"].str.contains("注釈 1"), "isFake"] = 0

    test[["id", "isFake"]].to_csv("submission.csv", index=False)
