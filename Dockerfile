# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

# ARG USER_UID
# ARG USER_GID

# FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
FROM tensorflow/tensorflow:latest-gpu-jupyter


# RUN addgroup --gid $USER_GID builder && \
#     adduser --uid $USER_UID --gid $USER_GID --disabled-password --gecos "" builder

# USER builder 

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

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
ENV PATH="/root/.local/bin:${PATH}"

# COPY . /code/
# RUN pip install --no-cache-dir -r requirements.txt

