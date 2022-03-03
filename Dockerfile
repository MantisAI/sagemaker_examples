FROM python:3.8.6-slim-buster

# Install Rust
RUN apt update -y
RUN apt install curl -y
RUN apt install apt build-essential -y
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install sagemaker-training
