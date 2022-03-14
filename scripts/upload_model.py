import sagemaker
import typer


def upload_model(local_model_path, s3_model_path):
    assert s3_model_path.startswith("s3://")
    bucket, key_prefix = s3_model_path[5:].split("/", 1)
    session = sagemaker.Session()
    session.upload_data(local_model_path, bucket=bucket, key_prefix=key_prefix)


if __name__ == "__main__":
    typer.run(upload_model)
