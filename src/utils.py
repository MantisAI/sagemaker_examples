import csv


def load_data(data_path):
    data = []
    with open(data_path, encoding="utf-8-sig") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            data.append((row["text"], row["label"]))
    X, y = zip(*data)
    return X, y
