export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
source /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/liruihan05/activate.sh sglang
MODEL_PATH="/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/liruihan05/models/open-source/Qwen3-235B-A22B"
python -m sglang.launch_server --model-path $MODEL_PATH --tp 8 --port 1690 --tool-call-parser qwen