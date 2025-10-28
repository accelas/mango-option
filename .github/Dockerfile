# Dockerfile for mango-iv CI environment
# This image pre-installs all dependencies to speed up CI runs
FROM debian:trixie

# Install all required dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        wget \
        ca-certificates \
        build-essential \
        python3 \
        systemtap-sdt-dev \
        libquantlib0-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
