FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    git \
    bzip2 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt