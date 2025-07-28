
# Dual-DP

Implementation of **Dual-Differential Privacy (Dual-DP)** applied to MiniGPT-4 for multimodal large language models. This project integrates **embedding-level noise (Stage 1)** and **parameter-level LoRA noise (Stage 2)** to achieve robust privacy-preserving training on image-text alignment datasets.

---

## Installation

### 1. Prepare the Code and Environment

Clone this repository, create a Conda environment, and activate it:

```bash
git clone https://github.com/oaphyapran365/Dual-DP-Multimodal-LLM.git
cd Dual-DP-Multimodal-LLM
conda env create -f environment.yml
conda activate minigptv
````

---

### 2. Prepare the Pretrained LLM Weights

This project supports **Vicuna V0** (7B and 13B) and **Llama 2 Chat 7B** backbones. Download the required LLM weights from Hugging Face (requires `git-lfs`):

| Vicuna 7B | Vicuna 13B | Llama 2 Chat 7B |
|--------------|---------------|-----------------|
| [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) |

After downloading, set the `llama_model` path in the model config file:

- **duelDP (Vicuna 7B or 13B):** Edit *duelDP_vicuna0.yaml* (Line 18)  
- **duelDP (Llama 2 7B):** Edit *duelDP_llama2.yaml* (Line 15)




### Resource Usage

* **8-bit mode by default** (beam width = 1)
* GPU memory required:

  * **13B model**: \~23 GB
  * **7B model**: \~11.5 GB
* For **16-bit mode** (higher quality), set `low_resource: False` in config:

  *[duelDP\_llama2\_eval.yaml]*
  *[duelDP\_eval.yaml]*

---

## Experimental Configuration

### Model Architecture

* **Visual Encoder:** EVA-CLIP-G
* **Q-Former:** BERT-based cross-attention

---

### Differential Privacy

* **Stage-1 (Embedding DP):** `sigma_embed = 0.5`
* **Stage-2 (Parameter DP via LoRA):** `sigma_proj = 1.0`, `clip_proj = 1.0`

---

### Training Hyperparameters

* Batch size: 12
* Learning rate: 3e-5 
* Weight decay: 0.05
* Epochs: 5
* Warmup steps: 200
* Image size: 224
* Max text length: 160 tokens

---


### Dataset

* **Training data:** `LAION and CC3M+CC12M+SBU` (image-text alignment)
* **Pre-processing:** BLIP2 image & caption processors

---

## Computing Infrastructure

### Hardware

* **GPU:** 4 × NVIDIA A100-SXM4 (80 GB each)
* **CPU:** Dual AMD EPYC 7742 (256 logical cores)
* **RAM:** 2 TB DDR4
* **Storage:** 41 TB shared NFS

---

### Software

* **OS:** Ubuntu 20.04 LTS, Kernel 5.15
* **Python:** 3.9
* **PyTorch:** 2.1.0 (CUDA 12.8)
* **Transformers:** 4.36.0
* Additional libraries: `timm`, `torchvision`, `einops`, `accelerate`, `gradio`

---

## Training and Evaluation

* Training details: [duelDP_train.md](duelDP_train.md)
* Finetuning/Evaluation: [eval\_readme.md](eval_scripts/eval\_readme.md)

---

## Acknowledgement

This work builds upon [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).


