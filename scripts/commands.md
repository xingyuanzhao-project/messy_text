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
module load cray-python/3.11.5
python --version
```

### Create venv and install
```bash
rm -rf venv
python -m venv venv
source venv/bin/activate
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

huggingface-cli login


huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

huggingface-cli download hugging-quants/gemma-2-9b-it-AWQ-INT4
huggingface-cli download solidrust/Mistral-7B-Instruct-v0.3-AWQ
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
huggingface-cli download openai/gpt-oss-20b
```


### Start vLLM server
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
source venv/bin/activate
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --port 8000 --host 0.0.0.0 --max-model-len 8192
```

### Run main.py (separate terminal)
```bash
cd /scratch/bbov/xzhao16/messy_text_server/
module purge
module load cray-python/3.11.5
source venv/bin/activate
python main.py
python scripts/run_processing.py
python scripts/run_evaluation.py
```

### Check Status
```bash
nvidia-smi
ps aux | grep main.py
watch -n 2 nvidia-smi
tail -f /scratch/bbov/xzhao16/messy_text_server/processing.log
```



### local ubuntu vm
```bash
nvidia-smi
ls -d */
source vllm_venv/bin/activate
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --quantization awq --port 8000 --host 0.0.0.0 --max-model-len 8192
```

### cursor terminal

```bash
.\.venv\Scripts\Activate.ps1
python scripts/run_processing.py
python scripts/run_evaluation.py
```