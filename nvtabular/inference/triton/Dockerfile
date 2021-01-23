ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.12-py3
FROM ${BASE_IMAGE}

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=/conda/bin:${PATH}
SHELL ["/bin/bash", "-c"]

# install cudf from conda
RUN conda install -c rapidsai -c nvidia -c numba -c conda-forge cudf=0.16 dask-cudf=0.16 nvtx cudatoolkit=11.0 

# python dependencies needed for triton
RUN pip3 install --upgrade grpcio-channelz  grpcio-tools dask 

# bundle up the nvtabular wheel and install
COPY nvtabular-0.3.0-py3-none-any.whl /tmp
RUN pip install /tmp/nvtabular-0.3.0-py3-none-any.whl
# RUN pip3 install git+https://github.com/benfred/nvtabular.git@triton_python_poc

