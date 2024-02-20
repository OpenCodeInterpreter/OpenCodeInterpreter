cd evaluation/evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd ../evaluate
gpu_start=0
MODEL_NAME_OR_PATHS=(
/ML-A800/home/zhengtianyu/code-interpreter/OpenCodeInterpreter-DS-6.7B
)
DATASET_TYPE="humaneval" #"mbpp"
EVAL_TYPE="plus" #base

for rank in {0..0}; do
    gpu_rank=$((rank + gpu_start))
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATHS[$rank]}"
    echo $MODEL_NAME_OR_PATH

    MODEL=$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|_|g')
    output_path="./human_feedback_output"

    LOG_DIR="$output_path/$MODEL"
    mkdir -p "${LOG_DIR}"
    ABS_LOG_DIR=$(realpath "$LOG_DIR")
    echo "Absolute path for mkdir: $ABS_LOG_DIR"

    LOG_FILE="gen_${DATASET_TYPE}_${EVAL_TYPE}_solution_multiround.log"

    CUDA_VISIBLE_DEVICES=$gpu_rank nohup python multi_turn/gen_plus_solution_multiround_human_feedback.py \
        --model "$MODEL_NAME_OR_PATH" \
        --output_path $output_path  \
        --log_file ${LOG_FILE} \
        --version $EVAL_TYPE \
        --dataset $DATASET_TYPE \
        --resume \
        > "${LOG_DIR}/${LOG_FILE}" 2>&1 &

done
