from collections import Counter
import argparse
import pickle
import csv
import os

from tqdm import tqdm
import torch


def load_data(data_path):
    data = []
    with open(data_path, encoding="utf-8-sig") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            data.append((row["text"], float(row["label"])))
    X, y = zip(*data)
    return X, y


class Tokenizer:
    def __init__(self, vocab_size, seq_len):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def fit(self, X):
        tokens = [token for x in tqdm(X) for token in x.split()]
        vocab = [token for token, _ in Counter(tokens).most_common()[: self.vocab_size]]
        self.word2id = {token: idx for idx, token in enumerate(vocab)}

    def transform(self, X):
        X_vec = torch.zeros(len(X), self.seq_len, dtype=torch.int32)
        for i, x in enumerate(tqdm(X)):
            for j, token in enumerate(x.split()):
                if j >= self.seq_len:
                    break
                X_vec[i, j] = self.word2id.get(token, 0)
        return X_vec


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
    vocab_size: int = 1000,
    seq_len: int = 100,
    emb_size: int = 100,
    hidden_size: int = 200,
    num_layers: int = 2,
):
    X, y = load_data(os.path.join(data_path, "movie.csv"))

    tokenizer = Tokenizer(vocab_size, seq_len)
    tokenizer.fit(X)
    X_vec = tokenizer.transform(X)
    y = torch.tensor(y)

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
    argparser.add_argument("--vocab_size", type=int, default=1000)
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
        args.vocab_size,
        args.seq_len,
        args.emb_size,
        args.hidden_size,
        args.num_layers,
    )
