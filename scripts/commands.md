# messy_text_server setup

## Session requirements
- 16 CPU
- 62GB RAM
- 1 GPU

---

## Check available Python modules

### List all Python modules
```bash
module avail python
```

### Check model cache
```bash
ls ~/.cache/huggingface/hub/
huggingface-cli scan-cache
```

### Check GPU
```bash
nvidia-smi
```

---

## Setup environment (cray-python/3.11.5)
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
module purge
module load cray-python/3.11.7
module load cuda
python --version
```

### Create venv and install
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
module purge
module load cray-python/3.11.7
module avail cuda
module load cuda/12.8
rm -rf venv
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
source venv/bin/activate
python -m pip install -r requirements.txt
pip freeze
```

### Check model cache
```bash
ln -s /scratch/bbov/xzhao16/huggingface_cache ~/.cache/huggingface
ls ~/.cache/huggingface/hub/
```

### Download model
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
source venv/bin/activate
pip install huggingface_hub
ls ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/*

huggingface-cli login

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
huggingface-cli download gaunernst/gemma-3-12b-it-int4-awq
huggingface-cli download mistralai/Ministral-3-8B-Instruct-2512
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
huggingface-cli download openai/gpt-oss-20b
huggingface-cli download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
# huggingface-cli download hugging-quants/gemma-2-27b-it-AWQ
huggingface-cli download pytorch/gemma-3-27b-it-AWQ-INT4
huggingface-cli download stelterlab/Mistral-Small-24B-Instruct-2501-AWQ

```


### Start vLLM server
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
source venv/bin/activate
module purge
module load cray-python/3.11.7
python --version
module load cuda
ls ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/*
/scratch/bbov/xzhao16/messy_text_server/venv/bin/pip show vllm

vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --port 8000 --host 0.0.0.0 --max-model-len 49152
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.90

vllm serve gaunernst/gemma-3-12b-it-int4-awq --port 8000 --host 0.0.0.0 --max-model-len 49152
vllm serve mistralai/Ministral-3-8B-Instruct-2512 --tokenizer_mode mistral --config_format mistral --load_format mistral --port 8000 --host 0.0.0.0 --max-model-len 49152

# ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/
# nano ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641/config.json
# vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
#   --quantization awq \
#   --port 8000 \
#   --host 0.0.0.0 \
#   --max-model-len 49152
# /scratch/bbov/xzhao16/messy_text_server/venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
#   --quantization awq \
#   --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
#   --port 8000 \
#   --host 0.0.0.0 \
#   --max-model-len 49152
# /scratch/bbov/xzhao16/messy_text_server/venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
#   --quantization awq \
#   --rope-scaling '{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":32768}' \
#   --port 8000 \
#   --host 0.0.0.0 \
#   --max-model-len 49152

vllm serve openai/gpt-oss-20b --port 8000 --host 0.0.0.0 --max-model-len 49152
vllm serve openai/gpt-oss-20b --async-scheduling --port 8000 --host 0.0.0.0 --max-model-len 49152

vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --tensor-parallel-size 4 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 50
```

### Run main.py (separate terminal)
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
module purge
module load cray-python/3.11.7
module load cuda
source venv/bin/activate

python scripts/run_summary.py
python scripts/run_summary_conversation_llama.py
python scripts/run_summary_conversation_gemma.py
python scripts/run_summary_conversation_mistral.py
python scripts/run_summary_conversation_gptoss.py
# python scripts/run_summary_conversation_qwen.py
# python scripts/run_summary_conversation_by_label.py
# sleep 3600 && python scripts/run_summary_conversation_by_label.py
python scripts/run_classification.py
python scripts/run_evaluation.py


```

### Check Status
```bash
nvidia-smi
ps aux | grep python
watch -n 2 nvidia-smi
nvidia-smi -l 2
tail -f /scratch/bbov/xzhao16/messy_text_server/processing.log
curl http://localhost:8000/metrics | grep vllm
```


# local setup
### local ubuntu vm
```bash
nvidia-smi
ls -d */
source vllm_venv/bin/activate
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --port 8000 --host 0.0.0.0 --max-model-len 49152
```

### cursor terminal

```bash
.\.venv\Scripts\Activate.ps1
python scripts/run_summary.py
python scripts/run_summary_conversation.py
python scripts/run_summary_conversation_by_label.py
python scripts/run_classification.py
python scripts/run_evaluation.py
```