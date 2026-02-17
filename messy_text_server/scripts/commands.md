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
pip install -r requirements.txt
pip freeze
```

### Check model cache
```bash
ls ~/.cache/huggingface/hub/
```

### Download model
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
pip install huggingface_hub
ls ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/*

huggingface-cli login

huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
huggingface-cli download gaunernst/gemma-3-12b-it-int4-awq
huggingface-cli download mistralai/Ministral-3-8B-Instruct-2512
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
huggingface-cli download openai/gpt-oss-20b
```


### Start vLLM server
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
source venv/bin/activate
module purge
module load cray-python/3.11.7
module load cuda
ls ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/*

vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --port 8000 --host 0.0.0.0 --max-model-len 49152

vllm serve gaunernst/gemma-3-12b-it-int4-awq --port 8000 --host 0.0.0.0 --max-model-len 49152
vllm serve mistralai/Ministral-3-8B-Instruct-2512 --tokenizer_mode mistral --config_format mistral --load_format mistral --port 8000 --host 0.0.0.0 --max-model-len 49152

ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/
nano ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641/config.json
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 49152

vllm serve openai/gpt-oss-20b --port 8000 --host 0.0.0.0 --max-model-len 49152
vllm serve openai/gpt-oss-20b --async-scheduling --port 8000 --host 0.0.0.0 --max-model-len 49152
```

### Run main.py (separate terminal)
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
module purge
module load cray-python/3.11.7
module load cuda
source venv/bin/activate
python scripts/run_summary.py
python scripts/run_summary_conversation.py
python scripts/run_classification.py
python scripts/run_evaluation.py

python scripts/run_summary_conversation.py && python scripts/run_classification.py && python scripts/run_evaluation.py
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
python scripts/run_classification.py
python scripts/run_evaluation.py
```