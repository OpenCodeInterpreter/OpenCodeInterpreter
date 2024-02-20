cd evaluation/evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd ../evaluate
MODEL_NAME_OR_PATHS=(
gpt3.5
# gpt4
)
DATASET_TYPES=(
# mbpp 
humaneval
)
EVAL_TYPES=(
base
# plus
)
for DATASET_TYPE in "${DATASET_TYPES[@]}"; do
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATHS[@]}"; do

            MODEL=$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|_|g')
            output_path="./execution_feedback_output"

            LOG_DIR="$output_path/$MODEL"
            mkdir -p "${LOG_DIR}"
            ABS_LOG_DIR=$(realpath "$LOG_DIR")
            echo "Absolute path for mkdir: $ABS_LOG_DIR"

            LOG_FILE="gen_${DATASET_TYPE}_${EVAL_TYPE}_solution_multiround.log"

            CUDA_VISIBLE_DEVICES=-1 nohup python multi_turn/gpt_gen_plus_solution_multiround.py \
                --model "$MODEL_NAME_OR_PATH" \
                --output_path $output_path  \
                --log_file ${LOG_FILE} \
                --version $EVAL_TYPE \
                --dataset $DATASET_TYPE \
                --resume \
                > "${LOG_DIR}/${LOG_FILE}" 2>&1 &
        done
    done
done



