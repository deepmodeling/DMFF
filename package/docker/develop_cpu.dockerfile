FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget -y && apt-get clean all

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda init bash && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda create -n dmff python=3.8 openmm=7.7.0 rdkit freud pytest openbabel biopandas mdtraj==1.9.7 -c conda-forge -y && \
    conda init && \
    source activate dmff && \
    pip install pymbar==4.0.1 && \
    pip install jaxlib[cpu]==0.3.15 && \
    pip install jax==0.3.15 && \
    pip install jax-md==0.2.0 && \
    pip install tqdm && \
    conda remove cudatoolkit --force -y && \
    conda clean --all -y && \
    rm -rf /root/.cache/pip && \
    echo "source activate dmff" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]