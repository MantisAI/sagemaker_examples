import sagemaker
import typer


def upload_data(local_data_path, s3_data_path):
    assert s3_data_path.startswith("s3://")
    bucket, key_prefix = s3_data_path[5:].split("/", 1)
    session = sagemaker.Session()
    session.upload_data(local_data_path, bucket=bucket, key_prefix=key_prefix)


if __name__ == "__main__":
    typer.run(upload_data)
