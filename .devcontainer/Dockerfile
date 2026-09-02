FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# System libraries Open3D needs at runtime, and the toolchain TEASER++ needs to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential libeigen3-dev libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# TEASER++ is the reference you check yourself against, and it is DELIBERATELY not built here.
# Building it is part of the lab (see the README), it takes several minutes, and a failed build
# must never block the rest of the work — `make reproduce` runs without it and skips one column.
