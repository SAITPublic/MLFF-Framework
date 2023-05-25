# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
 
# ------------------------------------------------------------
# Install python
# ------------------------------------------------------------
RUN apt-get update
RUN apt install python3.8 -y
RUN apt install python3-pip -y
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN python -m pip install pip --upgrade
RUN pip install virtualenv
RUN pip install setuptools --upgrade
RUN apt-get install python3.8-dev -y
 
 
# ------------------------------------------------------------
# Install packages
# ------------------------------------------------------------
RUN apt-get update -y
RUN apt-get -y install sudo
RUN apt-get -y install git
RUN apt-get install -y --no-install-recommends openssh-server
RUN apt update -y
RUN apt-get update -y
RUN apt install vim -y
RUN pip install -U setuptools
 
# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------
RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch_geometric
RUN pip install torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch_runstats
RUN pip install torch_ema
RUN pip install ase
RUN pip install matplotlib
RUN pip install numba
RUN pip install pymatgen
RUN pip install tensorboard
RUN pip install tqdm 
RUN pip install lmdb
RUN pip install submitit 
RUN pip install wandb  
RUN pip install e3nn
RUN pip install prettytable 
 
CMD [ "/bin/bash" ]
