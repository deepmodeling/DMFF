FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y wget cmake git g++

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda init bash && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN eval "$(/opt/miniconda/bin/conda shell.bash hook)" && \
	export CONDA_OVERRIDE_CUDA="12.0" && \
	conda create -y -n dmff_omm -c conda-forge python=3.11 openmm libtensorflow_cc tensorflow-gpu swig numpy && \
	conda activate dmff_omm && \
	TF_VERSION=$(python -c 'import tensorflow as tf; print(tf.__version__)') && \
	wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v$TF_VERSION.tar.gz && \
	tar -xf v$TF_VERSION.tar.gz && \
	mkdir -p ${CONDA_PREFIX}/include/tensorflow/c && \
	cp -r tensorflow-$TF_VERSION/tensorflow/c ${CONDA_PREFIX}/include/tensorflow && \
	rm -r tensorflow-$TF_VERSION v$TF_VERSION.tar.gz

# install TF C API for cppflow: https://www.tensorflow.org/install/lang_c
# wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$TF_VERSION.tar.gz
# tar -xf libtensorflow-gpu-linux-x86_64-$TF_VERSION.tar.gz -C /usr/local

RUN eval "$(/opt/miniconda/bin/conda shell.bash hook)" && \
	conda activate dmff_omm && \
	git clone https://github.com/caic99/DMFF.git && \
	git clone https://github.com/serizba/cppflow.git && \
		cd cppflow && \
		git apply ../DMFF/backend/openmm_dmff_plugin/tests/cppflow_empty_constructor.patch && \
		cp -r include/cppflow ${CONDA_PREFIX}/include && \
		cd .. && \
		rm -r cppflow && \
	export OPENMM_INSTALLED_DIR=$CONDA_PREFIX && \
	export CPPFLOW_INSTALLED_DIR=$CONDA_PREFIX && \
	export LIBTENSORFLOW_INSTALLED_DIR=$CONDA_PREFIX && \
	cd DMFF/backend/openmm_dmff_plugin/ && \
	mkdir build && cd build && \
	cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} \
		-DCPPFLOW_DIR=${CPPFLOW_INSTALLED_DIR} \
		-DTENSORFLOW_DIR=${LIBTENSORFLOW_INSTALLED_DIR} && \
	make -j && make install && \
	make -j PythonInstall && \
	cd / && rm -r DMFF
	# python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100 && \
