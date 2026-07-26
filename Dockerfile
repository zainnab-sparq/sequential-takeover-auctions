# Pinned base image (patch-level tag) so rebuilds are reproducible. For a fully
# content-addressed build, replace the tag with the digest printed by
# `docker buildx imagetools inspect python:3.12.13-slim-bookworm`.
FROM python:3.12.13-slim-bookworm

# libgomp1: OpenMP runtime some OpenSpiel ops rely on
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CPU-only PyTorch (unlocks OpenSpiel's pytorch agents: NFSP, PPO, DQN, PG).
# Pinned so deep-method numerics are reproducible across rebuilds.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Source is mounted at runtime; expose it on the path
ENV PYTHONPATH=/work/src
# Pin the CPU thread count so multithreaded BLAS/OpenMP float reductions are bounded
# and run-to-run variation is controlled (the experiment script also calls
# torch.set_num_threads to match). Override at runtime with -e OMP_NUM_THREADS=N.
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
CMD ["bash"]
