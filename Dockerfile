# Reproducible environment for the experiments in this paper.
#
# CPU-only on purpose: the padded probe (the discriminating experiment) runs in
# about 70 s per condition on CPU, so no GPU is needed to check the main claim.
#
#   docker build -t poslookup .
#   docker run --rm poslookup python code/padded.py --task copy --pe sin --steps 2000
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/    code/
COPY results/ results/
COPY figures/ figures/
COPY paper/   paper/

WORKDIR /work/code

# Reproduce the discriminating experiment by default.
CMD ["python", "padded.py", "--task", "copy", "--pe", "sin", "--steps", "2000"]
