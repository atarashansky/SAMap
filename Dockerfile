FROM debian:buster-slim
ARG PATH="/root/miniconda/bin:${PATH}"
ENV PATH="/root/miniconda/bin:${PATH}"

RUN apt-get update \
        && apt-get install -y wget \
        && apt-get install -y g++ \ 
        && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash /tmp/miniconda.sh -b -p /root/miniconda \
    && rm -f /tmp/miniconda.sh \
    && conda update -n base -c defaults conda \ 
    && conda install pip \ 
    && conda clean -afy

RUN /root/miniconda/bin/pip install hnswlib==0.4.0

COPY --from=scrna:miniconda.base /root/miniconda /root/miniconda
ARG PATH="/root/miniconda/bin:${PATH}"
ENV PATH="/root/miniconda/bin:${PATH}"
# installing base dependencies
RUN conda install -c plotly -c conda-forge -c bioconda \ 
			    jupyter==1.0.0 \ 
                            plotly==4.0.0 \ 
                            colorlover==0.3.0 \ 
                            ipyevents==0.8.1 \ 
                            numpy==1.19.2 \ 
                            scipy==1.5.2 \ 
                            python-igraph==0.8.3 \ 
                            leidenalg==0.8.3 \
                            pandas==1.0.0 \
                            scanpy==1.6.0 \ 
                            biopython \ 
                            umap-learn==0.4.6 && conda clean -afy 
                            

# copying over your git folders and installing them.
COPY git_repos/self-assembling-manifold /tmp/self-assembling-manifold
RUN pip install /tmp/self-assembling-manifold/. --no-dependencies && rm -rf /tmp/ && rm -rf ~/.cache
COPY git_repos/SAMap /tmp/SAMap
RUN pip install /tmp/SAMap/. --no-dependencies && rm -rf /tmp/ && rm -rf ~/.cache

# creating file
RUN mkdir /jupyter
RUN mkdir /jupyter/notebooks
WORKDIR /jupyter/
CMD jupyter notebook --port=$PORT --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="" --NotebookApp.token=""