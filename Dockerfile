FROM ubuntu:24.04

ENV PATH=/root/.cargo/bin:$PATH
ENV MAKEFLAGS=-j$(($(nproc)-2))
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and tools
RUN apt update && \
	apt install -y git curl build-essential gcc-riscv64-linux-gnu
# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

WORKDIR /workspaces/jit-assembler