FROM continuumio/miniconda3

# Create the environment:
COPY requirements.txt .
RUN conda update conda
RUN conda install -c anaconda python=3.8.8
RUN conda config --env --add channels conda-forge
RUN conda install pytorch==1.9.0 torchvision==0.10.0 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda install torchmetrics==0.11.4
RUN conda install matplotlib==3.3.4
RUN conda install nibabel==3.2.1
RUN conda install nilearn==0.8.1
RUN conda install numpy==1.18.5
RUN conda install pandas==1.2.4
RUN conda install scikit-learn==1.0.2
RUN conda install scikit-image==0.19.2
RUN conda install torch-fidelity==0.3.0

RUN apt-get update
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Initialize the directories
RUN mkdir /PACGAN
RUN mkdir /PACGAN/Preprocessing
RUN mkdir /PACGAN/Training

# Copy the files
COPY ./Training/config.json /PACGAN/Training/config.json
COPY ./Training/assessment.py /PACGAN/Training/assessment.py
COPY ./Training/Discriminator.py /PACGAN/Training/Discriminator.py
COPY ./Training/generate_images.py /PACGAN/Training/generate_images.py
COPY ./Training/Generator.py /PACGAN/Training/Generator.py
COPY ./Training/Load_dataset.py /PACGAN/Training/Load_dataset.py
COPY ./Training/main.py /PACGAN/Training/main.py
COPY ./Training/model.py /PACGAN/Training/model.py
COPY ./Training/train.py /PACGAN/Training/train.py
COPY ./Training/utils.py /PACGAN/Training/utils.py
COPY ./Preprocessing/ADvsHC_matching.py /PACGAN/Preprocessing/ADvsHC_matching.py
COPY ./Preprocessing/divide_TrainTest.py /PACGAN/Preprocessing/divide_TrainTest.py

# Set the working directory
WORKDIR /PACGAN