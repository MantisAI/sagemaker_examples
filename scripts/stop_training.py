import sagemaker
import typer


def stop_training(job_name):
    session = sagemaker.Session()
    session.stop_training_job(job_name)

if __name__ == "__main__":
    typer.run(stop_training)
