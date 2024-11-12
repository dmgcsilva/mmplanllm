
cd ~/mmplanllm_release # change dir to where we want to run scripts

CKPT_DIR=$1
TEST_FILE="/path/to/your/data/mmplanllm_test_alternative_user.json"
TEST_FILE="/data/dmgc.silva/datasets/mmplanllm_dataset/mmplanllm_dataset_test.json"

FILENAME=$(basename "$CKPT_DIR")
FILENAME=$(basename "$FILENAME")
echo $FILENAME

echo "Using base model: $BASE_MODEL"
echo "Using checkpoint: $CKPT_DIR"

python src/vmr_inference.py \
    --base_model llama2 \
    --test_file $TEST_FILE \
    --ckpt_path $CKPT_DIR \
    --decoding_strategy greedy \
    --score_metrics \
    --max_new_tokens 32 \
    --max_samples 50000000 \
    --log_metrics True \
    --dataset_kwargs "{'feature_extractor_model': 'openai/clip-vit-large-patch14', 'context_size': 4, 'max_len':2048, 'only_visual': True}" \
    --dataset_type mmplanllm_dataset \
    --fp32 True \
    --inference_batch_size 4
