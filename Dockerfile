FROM ubuntu:24.04

ENV PATH=/root/.cargo/bin:$PATH
ENV MAKEFLAGS=-j$(($(nproc)-2))
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and tools
# Install RISC-V cross-compiler only on non-RISC-V platforms
RUN apt update && \
	apt install -y git curl build-essential && \
	if [ "$(uname -m)" != "riscv64" ]; then \
		apt install -y gcc-riscv64-linux-gnu; \
	fi

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

WORKDIR /workspaces/jit-assembler