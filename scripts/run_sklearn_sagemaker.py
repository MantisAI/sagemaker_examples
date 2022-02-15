from sagemaker.sklearn import SKLearn
import sagemaker
import typer


def run_sklearn_sagemaker(data_path, instance_type, role, min_df:int=5, max_ngram:int=1, stopwords="english", loss="hinge", learning_rate:float=1e-4):
    hyperparameters = {
        "min_df": min_df,
        "max_ngram": max_ngram,
        "stopwords": english,
        "loss": loss,
        "learning_rate": learning_rate
    }
    sk = SKLearn(
        entry_point="train_sklearn.py",
        framework_version="0.20.0",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        hyperparameters=hyperparameters
    )
    if instance_type == "local":
        data_path = f"file://{data_path}"
    else:
        assert "s3://" in data_path 
    sk.fit({"train": data_path})

if __name__ == "__main__":
    typer.run(run_sklearn_sagemaker)
