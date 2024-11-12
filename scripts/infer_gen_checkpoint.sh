
cd ~/mmplanllm_release # change dir to where we want to run scripts

CKPT_DIR=$1
TEST_FILE="/path/to/your/data"
TEST_FILE="/data/dmgc.silva/datasets/plangpt_dataset_2.4/plangpt_dataset_2.4_test.json"

FILENAME=$(basename "$CKPT_DIR")
FILENAME=$(basename "$FILENAME")
echo $FILENAME

echo "Using base model: $BASE_MODEL"
echo "Using checkpoint: $CKPT_DIR"

python src/text_inference.py \
    --base_model llama2 \
    --test_file $TEST_FILE \
    --ckpt_path $CKPT_DIR \
    --decoding_strategy greedy \
    --score_metrics \
    --max_new_tokens 128 \
    --max_samples 500000 \
    --log_metrics True \
    --dataset_kwargs "{'context_size': 4, 'max_len': 2048}" \
    --dataset_type mmplanllm_dataset \
    --fp32 True \
    --inference_batch_size 4
