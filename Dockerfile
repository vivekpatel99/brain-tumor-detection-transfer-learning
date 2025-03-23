# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter


FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3



WORKDIR /code

RUN apt-get update -y && \
    apt-get upgrade -y  \
    && apt-get install curl ffmpeg libsm6 libxext6  -y 


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV="/code/.venv" 
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml /code/
RUN uv sync --active 


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
