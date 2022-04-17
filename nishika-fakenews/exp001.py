import os

import datasets
import numpy as np
import pandas as pd
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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
    df.columns = ["id", "label", "text"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(cv.split(df, df["label"])):
        df.loc[val_index, "fold"] = int(n)
    df["fold"] = df["fold"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-large-japanese")

    test = pd.read_csv("../input/nishika-fakenews/test.csv")
    test_dataset = datasets.Dataset.from_pandas(test[["text"]])
    test_tokenized = test_dataset.map(preprocess_function, batched=True)

    for fold_id in range(20):

        if fold_id in (0,):

            train_dataset = datasets.Dataset.from_pandas(
                df.query(f"fold != {fold_id}")[["text", "label"]]
            )
            train_tokenized = train_dataset.map(preprocess_function, batched=True)

            val_dataset = datasets.Dataset.from_pandas(
                df.query(f"fold == {fold_id}")[["text", "label"]]
            )
            val_tokenized = val_dataset.map(preprocess_function, batched=True)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(
                "cl-tohoku/bert-large-japanese", num_labels=2
            )

            training_args = TrainingArguments(
                output_dir=f"../tmp/results{fold_id}",
                learning_rate=1e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=64,
                num_train_epochs=3,
                weight_decay=0.01,
                evaluation_strategy="steps",
                eval_steps=250,
                #                 load_best_model_at_end=True,
                #                 save_steps=1000,
                gradient_accumulation_steps=2,
                #                 save_total_limit=3,
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
            results = trainer.predict(test_dataset=test_tokenized)
            np.save(f"test_prediction{fold_id}", results.predictions)

    test["isFake"] = np.argmax(results.predictions, axis=-1)
    test[["id", "isFake"]].to_csv("submission.csv", index=False)
