FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda11.6.1
USER root
RUN apt update && apt install ssh -y

RUN apt-get update
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y vim-common

# authorize SSH connection with root account
RUN sed -i 's/#\s*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# change root password to 'mypassword' <-- PICK SOME OTHER PASSWORD HERE
RUN echo "root:1234567" | chpasswd

# Necessary configuration for sshd
RUN mkdir /var/run/sshd

# COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
# RUN pip install -r /requirements.txt
RUN pip install jupyterlab

WORKDIR /

# Open Jupyter lab
ENV JUPYTER_TOKEN=1234567 

# Start sshd at container spin-up time
CMD service ssh start && bash
