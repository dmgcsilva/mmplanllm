
cd ~/mmplanllm_release # change dir to where we want to run scripts

CKPT_DIR=$1

sh scripts/infer_vsg_checkpoint.sh $CKPT_DIR
sh scripts/infer_vmr_checkpoint.sh $CKPT_DIR
sh scripts/infer_gen_checkpoint.sh $CKPT_DIR
