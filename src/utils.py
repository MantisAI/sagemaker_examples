import csv


def load_data(data_path):
    data = []
    with open(data_path, encoding="utf-8-sig") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            data.append((row["text"], float(row["label"])))
    X, y = zip(*data)
    return list(X), list(y)
