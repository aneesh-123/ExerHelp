#!/usr/bin/env bash
# ------------------------- POSE MODELS -------------------------
# Downloading the pose-model trained on COCO
echo "Hi Aneesh"
COCO_POSE_URL="https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel"
COCO_FOLDER="pose/coco/"
wget -c ${COCO_POSE_URL} -P ${COCO_FOLDER}
echo "Hi Rohan"
# Downloading the pose-model trained on MPI
MPI_POSE_URL="https://www.dropbox.com/s/drumc6dzllfed16/pose_iter_160000.caffemodel"
MPI_FOLDER="pose/mpi/"
wget -c ${MPI_POSE_URL} -P ${MPI_FOLDER}
