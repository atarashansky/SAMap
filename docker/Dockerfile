FROM debian:buster-slim
ARG PATH="/root/miniconda/bin:${PATH}"
ENV PATH="/root/miniconda/bin:${PATH}"

RUN apt-get update \
        && apt-get install -y wget \
        && apt-get install -y g++ \ 
        && apt-get install -y build-essential \
	&& apt-get install -y git \ 
        && apt-get install -y libcurl4-gnutls-dev libxml2-dev libssl-dev \ 
        && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash /tmp/miniconda.sh -b -p /root/miniconda \
    && rm -f /tmp/miniconda.sh \
    && conda update -n base -c defaults conda \ 
    && conda install python=3.8 pip \ 
    && conda clean -afy

RUN conda install -c plotly -c conda-forge -c bioconda -c anaconda \ 
			                jupyter==1.0.0 \ 
                            plotly==4.0.0 \ 
                            colorlover==0.3.0 \ 
                            ipyevents==0.8.1 \ 
                            numpy==1.19.2 \ 
                            scipy==1.5.2 \
                            numba==0.51.2 \  
                            dill \ 
                            python-igraph==0.8.3 \ 
                            leidenalg==0.8.3 \
                            pandas==1.0.0 \
                            scanpy==1.6.0 \ 
                            biopython \ 
                            umap-learn==0.4.6 \ 
                            && conda clean -afy 


RUN wget "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.9.0/ncbi-blast-2.9.0+-x64-linux.tar.gz" && \ 
    tar -xzvf "ncbi-blast-2.9.0+-x64-linux.tar.gz" \
        -C "/root/miniconda/bin/" \
        --strip-components=2 \
        "ncbi-blast-2.9.0+/bin/"
    
COPY . /tmp/SAMap
RUN /root/miniconda/bin/pip install /tmp/SAMap/. && rm -rf ~/.cache
RUN /root/miniconda/bin/pip install hnswlib holoviews bokeh==2.2.3 && rm -rf ~/.cache
RUN /root/miniconda/bin/pip install git+https://github.com/eggnogdb/eggnog-mapper.git

RUN chmod ugo+rwx /root
RUN chmod ugo+rwx /tmp && mkdir /jupyter && mkdir /jupyter/notebooks
WORKDIR /jupyter/

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

CMD jupyter notebook --port=$PORT --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="" --NotebookApp.token=""
