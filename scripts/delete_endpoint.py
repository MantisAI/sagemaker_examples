import sagemaker
import typer


def delete_endpoint(endpoint_name):
    session = sagemaker.Session()
    session.delete_endpoint(endpoint_name)

if __name__ == "__main__":
    typer.run(delete_endpoint)
