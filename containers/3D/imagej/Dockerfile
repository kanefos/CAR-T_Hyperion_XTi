# Micromamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update
RUN apt-get install -y wget unzip > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
RUN micromamba install -y -n base -c conda-forge \
        python=3.8\
        pyimagej  \
        openjdk=11 && \
    micromamba clean --all --yes
ENV JAVA_HOME="/usr/local"
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  
# Retrieve ImageJ and source code
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip &> /dev/null
RUN unzip fiji-linux64.zip > /dev/null
RUN rm fiji-linux64.zip

# set working directory
WORKDIR /fiji

# Copy the script in the folder
COPY ./save_single_channel.py /usr/local/bin/save_single_channel.py

# # Run the script
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python"]
# CMD ["python", "/usr/local/bin/save_single_channel.py"]
