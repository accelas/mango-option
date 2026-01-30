# SPDX-License-Identifier: MIT
# Dockerfile for mango-option CI environment
# This image pre-installs all dependencies to speed up CI runs
# Uses trixie-slim base for smaller size (~40% reduction)
FROM debian:trixie-slim

# Install all required dependencies
# Using --no-install-recommends to minimize image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates \
        build-essential \
        python3 \
        systemtap-sdt-dev \
        libquantlib0-dev \
        liblapacke-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
