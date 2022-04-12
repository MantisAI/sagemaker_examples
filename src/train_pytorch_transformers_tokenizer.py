from collections import Counter
import argparse
import pickle
import csv
import os

from transformers import AutoTokenizer
from tqdm import tqdm
import torch

from utils import load_data


class Model(torch.nn.Module):
    def __init__(self, vocab_size, seq_len, emb_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.bilstm = torch.nn.LSTM(
            emb_size, hidden_size, num_layers, bidirectional=True, batch_first=True
        )
        self.linear = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = torch.squeeze(x[:, -1, :])
        x = self.linear(x)
        x = torch.sigmoid(x)
        return torch.squeeze(x)


def train_pytorch(
    data_path,
    model_path,
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    seq_len: int = 100,
    emb_size: int = 100,
    hidden_size: int = 200,
    num_layers: int = 2,
):
    X, y = load_data(os.path.join(data_path, "movie.csv"))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    X_vec = torch.tensor(
        tokenizer(list(X), padding=True, truncation=True, max_length=seq_len)[
            "input_ids"
        ]
    )
    y = torch.tensor(y)

    vocab_size = X_vec.max() + 1
    seq_len = X_vec.shape[1]

    dataset = list(zip(X_vec, y))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(vocab_size, seq_len, emb_size, hidden_size, num_layers)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for example in tqdm(data):
            inputs, labels = example[0].to(device), example[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

    torch.save(model, os.path.join(model_path, "model.pt"))
    with open(os.path.join(model_path, "tokenizer.pkl"), "wb") as f:
        f.write(pickle.dumps(tokenizer))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", default=os.environ.get("SM_CHANNEL_TRAIN"))
    argparser.add_argument("--model_path", default=os.environ.get("SM_MODEL_DIR"))
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--learning_rate", type=float, default=1e-3)
    argparser.add_argument("--seq_len", type=int, default=100)
    argparser.add_argument("--emb_size", type=int, default=100)
    argparser.add_argument("--hidden_size", type=int, default=200)
    argparser.add_argument("--num_layers", type=int, default=2)
    args = argparser.parse_args()

    train_pytorch(
        args.data_path,
        args.model_path,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.seq_len,
        args.emb_size,
        args.hidden_size,
        args.num_layers,
    )
