FROM jupyter/datascience-notebook

USER root
RUN apt update
RUN pip install anndata umap-learn leidenalg igraph 
RUN apt install -y graphviz
RUN pip install seaborn --upgrade
RUN pip install xgboost 
RUN pip install scanpy squidpy "spatialdata[extra]" "napari[all]"
RUN pip install openchord
RUN pip install opencv-python

USER foster
WORKDIR /home/unix/foster
