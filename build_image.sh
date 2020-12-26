#!/bin/bash
read -e -p "Docker image name: " image
docker build . -f Dockerfile -t $image --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
