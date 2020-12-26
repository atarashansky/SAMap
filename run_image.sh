#!/bin/bash
read -e -p "Docker image name: " image
read -e -p "Docker container name: " name
read -e -p "Volume name: " folder
read -e -p "Jupyter notebook port: " port

path="${folder/#\~/$HOME}"
parentdir="$(dirname "$path")"
chmod 755 $parentdir

docker run -d --rm --name=$name \
           -v $path:/jupyter/notebooks \
           -p $port:$port -e PORT=$port $image
