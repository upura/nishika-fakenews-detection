import numpy as np
import pandas as pd

if __name__ == "__main__":
    pred0 = np.load("/kaggle/input/nishika-fakenews-pred/test_prediction_fold0.npy")
    pred1 = np.load("/kaggle/input/nishika-fakenews-pred/test_prediction_fold1.npy")
    pred2 = np.load("/kaggle/input/nishika-fakenews-pred/test_prediction_fold2.npy")
    pred3 = np.load("/kaggle/input/nishika-fakenews-pred/test_prediction_fold3.npy")
    pred4 = np.load("/kaggle/input/nishika-fakenews-pred/test_prediction_fold4.npy")
    results = (pred0 + pred1 + pred2 + pred3 + pred4) / 5
    test = pd.read_csv("/kaggle/input/nishika-fakenews/test.csv")
    test["isFake"] = np.argmax(results, axis=-1)
    test[["id", "isFake"]].to_csv("submission.csv", index=False)
