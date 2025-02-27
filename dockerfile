FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y curl wget git vim gcc make cmake python3 python3-pip
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir "setuptools<71.0.0" tensorboard
# RUN pip3 install --no-cache-dir flash-attn
RUN pip3 install --no-cache-dir -U transformers==4.48.1 datasets==3.1.0 accelerate==1.3.0 hf-transfer==0.1.9 deepspeed==0.15.4 trl==0.14.0 vllm==0.7.0 peft==0.14.0 bitsandbytes==0.45.0