#!/bin/sh
#SBATCH --partition=debug-lp

set -eux

TAG=diplomka_image
TMP_DIR=`mktemp -d`
CONTEXT_DIR=`mktemp -d`
IMG_DIR='imgdir2'

mkdir -p "$IMG_DIR" "$CONTEXT_DIR"

cp requirements.txt Dockerfile "$CONTEXT_DIR"

cd "$CONTEXT_DIR"
ch-build -t "$TAG" .
cd -

ch-builder2tar --nocompress "$TAG" "$TMP_DIR"
ch-tar2dir "${TMP_DIR}/${TAG}.tar" "$IMG_DIR"

rm -rf "$TMP_DIR" "$CONTEXT_DIR"