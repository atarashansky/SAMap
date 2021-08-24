FROM debian:buster-slim
RUN echo "Reset"
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

COPY . /tmp/SAMap
RUN /root/miniconda/bin/pip install -r /tmp/SAMap/docker/reqlist.txt
RUN pip install /tmp/SAMap/. --no-dependencies
RUN conda install nodejs -y
RUN pip install jupyterlab
RUN jupyter labextension install jupyterlab-plotly --no-build
RUN jupyter labextension install plotlywidget --no-build
RUN jupyter lab build --dev-build=False --minimize=False
RUN /root/miniconda/bin/pip install git+https://github.com/eggnogdb/eggnog-mapper.git
RUN rm -rf ~/.cache

RUN wget "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.9.0/ncbi-blast-2.9.0+-x64-linux.tar.gz" && \ 
    tar -xzvf "ncbi-blast-2.9.0+-x64-linux.tar.gz" \
        -C "/root/miniconda/bin/" \
        --strip-components=2 \
        "ncbi-blast-2.9.0+/bin/"
    

RUN chmod ugo+rwx /root
RUN chmod ugo+rwx /tmp && mkdir /jupyter && mkdir /jupyter/notebooks
WORKDIR /jupyter/

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

CMD jupyter lab --port=$PORT --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="" --NotebookApp.token=""
