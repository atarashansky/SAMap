#!/bin/bash
read -e -p "Docker container name: " name
read -e -p "Mount volume path: " folder
read -e -p "Jupyter notebook port: " port

image=tarashan/samap:latest
path="${folder/#\~/$HOME}"
parentdir="$(dirname "$path")"
chmod 755 $parentdir

docker run -d --rm --name=$name \
           -v $path:/jupyter/notebooks \
           -p $port:$port -e PORT=$port $image
