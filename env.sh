conda activate h_face

# cuda path
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="${PYTHONPATH}:/home/egseo/hugging_face/transformers/src/transformers/bfp_training"
export BFP_HOME="/home/egseo/hugging_face/transformers/src/transformers/bfp_training"