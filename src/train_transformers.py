import argparse
import csv
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollator,
)
import torch


def load_data(data_path):
    data = []
    with open(data_path, encoding="utf-8-sig") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            data.append((row["text"], row["label"]))
    data = data[:10]
    X, y = zip(*data)
    return X, y


def transform_data(data, tokenizer):
    X, y = zip(*data)
    input_ids = tokenizer(list(X), truncation=True, padding=True)["input_ids"]
    for i in range(len(input_ids)):
        yield {"input_ids": input_ids[i], "label": int(y[i])}


def train_transformers(
    data_path,
    model_path,
    pretrained_model="bert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
    weight_decay: float = 0.01,
):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    X, y = load_data(os.path.join(data_path, "movie.csv"))

    train_dataset = list(transform_data(zip(X, y), tokenizer))

    training_args = TrainingArguments(
        output_dir="./",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    model.save_pretrained(model_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", default=os.environ.get("SM_CHANNEL_TRAIN"))
    argparser.add_argument("--model_path", default=os.environ.get("SM_MODEL_DIR"))
    argparser.add_argument("--pretrained_model", default="bert-base-uncased")
    argparser.add_argument("--learning_rate", type=float, default=2e-5)
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--weight_decay", type=float, default=0.01)
    args = argparser.parse_args()

    train_transformers(
        args.data_path,
        args.model_path,
        args.pretrained_model,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.weight_decay,
    )
