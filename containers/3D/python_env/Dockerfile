FROM jupyter/datascience-notebook

USER root
RUN apt update
RUN pip install anndata umap-learn leidenalg igraph 
RUN apt install -y graphviz
RUN pip install seaborn --upgrade
RUN pip install xgboost 
RUN pip install scanpy squidpy "spatialdata[extra]" "napari[all]"
RUN pip install imctools readimc
RUN apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0

RUN pip install cupy-cuda12x


RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cuml-cu12==24.6.* cucim-cu12==24.6.*

RUN pip install filetype
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
USER foster
WORKDIR /home/unix/foster
