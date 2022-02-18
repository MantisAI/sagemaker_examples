import sagemaker
import typer


def download_model(model_path, bucket, key_prefix="models"):
    session = sagemaker.Session()
    session.download_data(model_path, bucket, key_prefix)

if __name__ == "__main__":
    typer.run(download_model)

