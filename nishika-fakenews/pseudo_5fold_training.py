import os

import datasets
import numpy as np
import pandas as pd
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    os.environ["WANDB_DISABLED"] = "true"
    metric = load_metric("accuracy")

    df = pd.read_csv("../input/nishika-fakenews/train.csv")
    df.text = [
        data[: data.find("。")] + "。[SEP]" + data[data.find("。") + 1 :]
        for data in df.text
    ]
    df.columns = ["id", "label", "text"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(cv.split(df, df["label"])):
        df.loc[val_index, "fold"] = int(n)
    df["fold"] = df["fold"].astype(int)

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    test = pd.read_csv("../input/nishika-fakenews/test.csv")
    test.text = [
        data[: data.find("。")] + "。[SEP]" + data[data.find("。") + 1 :]
        for data in test.text
    ]
    test_dataset = datasets.Dataset.from_pandas(test[["text"]])
    test_tokenized = test_dataset.map(preprocess_function, batched=True)

    test_prediction0 = np.load(
        "../input/nishika-fakenews-rinna-japanese-gpt2-full/test_prediction0.npy"
    )
    test["pred0"] = test_prediction0[:, 0]
    test["pred1"] = test_prediction0[:, 1]
    test["label"] = (test["pred0"] < test["pred1"]).astype(int)
    test_filter = test.query("pred0 > 3 or pred0 < -3")[["id", "label", "text"]]

    for fold_id in range(5):

        if fold_id in (0, 1, 2, 3, 4):

            train_dataset = datasets.Dataset.from_pandas(
                pd.concat([df.query(f"fold != {fold_id}"), test_filter]).reset_index(
                    drop=True
                )[["text", "label"]]
            )
            train_tokenized = train_dataset.map(preprocess_function, batched=True)

            val_dataset = datasets.Dataset.from_pandas(
                df.query(f"fold == {fold_id}")[["text", "label"]]
            )
            val_tokenized = val_dataset.map(preprocess_function, batched=True)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(
                "rinna/japanese-gpt2-medium", num_labels=2
            )
            model.config.pad_token_id = model.config.eos_token_id

            training_args = TrainingArguments(
                output_dir=f"../tmp/results{fold_id}",
                learning_rate=4e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=64,
                num_train_epochs=6,
                weight_decay=0.01,
                evaluation_strategy="steps",
                eval_steps=250,
                load_best_model_at_end=True,
                save_steps=1000,
                gradient_accumulation_steps=3,
                save_total_limit=3,
                fp16=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            oof_results = trainer.predict(test_dataset=val_tokenized)
            np.save(f"oof_prediction{fold_id}", oof_results.predictions)
            oof = df.query(f"fold == {fold_id}").copy()
            oof["pred0"] = oof_results.predictions[:, 0]
            oof["pred1"] = oof_results.predictions[:, 1]
            oof.to_csv(f"oof{fold_id}.csv", index=False)

            results = trainer.predict(test_dataset=test_tokenized)
            np.save(f"test_prediction{fold_id}", results.predictions)
