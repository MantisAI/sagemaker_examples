import argparse
import pickle
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def load_data(data_path):
    data = []
    with open(data_path, encoding="utf-8-sig") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            data.append((row["text"], row["label"]))
    X, y = zip(*data)
    return X, y

def train(data_path, model_path, min_df:int=5, max_ngrams:int=1, stopwords="english", loss="hinge", learning_rate:float=1e-4):
    X, y = load_data(os.path.join(data_path, "movie.csv"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=min_df,
            ngram_range=(1, max_ngrams),
            stop_words=stopwords
        )),
        ("svm", SGDClassifier(
            loss=loss,
            alpha=learning_rate
        ))
    ])
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        f.write(pickle.dumps(model))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    argparser.add_argument("--model_path", type=str, default=os.environ.get("SM_MODEL_DIR"))
    argparser.add_argument("--min_df", type=int, default=5)
    argparser.add_argument("--max_ngrams", type=int, default=1)
    argparser.add_argument("--stopwords", type=str, default="english")
    argparser.add_argument("--loss", type=str, default="hinge")
    argparser.add_argument("--learning_rate", type=float, default=1e-4)
    args = argparser.parse_args()

    train(
        args.data_path,
        args.model_path,
        args.min_df,
        args.max_ngrams,
        args.stopwords,
        args.loss,
        args.learning_rate
    )
