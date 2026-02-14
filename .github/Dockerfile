# Shared CI environment for Bazel C++ projects
# Used by: mango-option, dbn-pipe
FROM debian:trixie-slim

# Install all required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates \
        build-essential \
        clang \
        python3 \
        python3-numpy \
        python3-dev \
        systemtap-sdt-dev \
        libquantlib0-dev \
        liblapacke-dev \
        libssl-dev \
        libzstd-dev \
        libpq-dev && \
    # Add Apache Arrow repository and install
    wget -q https://apache.jfrog.io/artifactory/arrow/debian/apache-arrow-apt-source-latest-trixie.deb && \
    apt-get install -y ./apache-arrow-apt-source-latest-trixie.deb && \
    rm apache-arrow-apt-source-latest-trixie.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libarrow-dev && \
    # Install Bazelisk
    wget -q https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

CMD ["/bin/bash"]
