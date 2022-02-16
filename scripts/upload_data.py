import sagemaker
import typer


def upload_data(data_path, bucket, key_prefix="data"):
    session = sagemaker.Session()
    session.upload_data(data_path, bucket=bucket, key_prefix=key_prefix)

if __name__ == "__main__":
    typer.run(upload_data)
