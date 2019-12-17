FROM ubuntu:16.04

RUN apt update && \
    apt upgrade -y &&\
    apt install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/dpovolotskiy/machine_learning_thesis

WORKDIR /machine_learning_thesis/

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade setuptools &&\
    pip3 install idna==2.5 --ignore-installed &&\
    pip3 install -r requirements.txt

RUN /bin/bash -c "wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

RUN echo "alias image_capt='python3.5 /machine_learning_thesis/main.py --mode predict --modelpath /machine_learning_thesis/my_model_15.h5 --filepath'" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
