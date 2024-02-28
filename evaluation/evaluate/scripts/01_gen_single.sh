cd evaluation/evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd ../evaluate
gpu_start=0
MODEL_NAME_OR_PATHS=(
/ML-A800/home/zhengtianyu/code-interpreter/OpenCodeInterpreter-DS-6.7B
)
DATASET_TYPES=(
humaneval
mbpp
)

i=0
for MODEL_NAME_OR_PATH in "${MODEL_NAME_OR_PATHS[@]}"; do
    for DATASET_TYPE in "${DATASET_TYPES[@]}"; do
        gpu_rank=$((i + gpu_start))
        MODEL=$(echo "$MODEL_NAME_OR_PATH" | sed 's|/|_|g')

        output_path="./output"
        LOG_DIR="$output_path/$MODEL"
        mkdir -p "${LOG_DIR}"
        ABS_LOG_DIR=$(realpath "$LOG_DIR")
        LOG_FILE=single_gen_${DATASET_TYPE}_plus_solution_singleround.log

        CUDA_VISIBLE_DEVICES=$gpu_rank nohup python single_turn/gen_plus_solution_singleround.py \
            --model "$MODEL_NAME_OR_PATH" \
            --output_path $output_path  \
            --log_file ${LOG_FILE} \
            --dataset $DATASET_TYPE \
            > "${LOG_DIR}/${LOG_FILE}" 2>&1 &
        ((i++))
done
done
