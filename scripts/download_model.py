import os

import sagemaker
import typer


def download_model(local_model_path, s3_model_path, job_name=None):
    assert s3_model_path.startswith("s3://")
    bucket, key_prefix = s3_model_path[5:].split("/", 1)

    if job_name:
        key_prefix = os.path.join(key_prefix, job_name, "output")
        print(key_prefix)

    session = sagemaker.Session()
    session.download_data(local_model_path, bucket, key_prefix)


if __name__ == "__main__":
    typer.run(download_model)
