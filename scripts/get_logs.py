import sagemaker
import typer


def get_logs(job_name, wait:bool=True):
    session = sagemaker.Session()
    session.logs_for_job(job_name, wait)

if __name__ == "__main__":
    typer.run(get_logs)
