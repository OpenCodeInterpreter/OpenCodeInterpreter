cd evaluation/evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)

directory="../evaluate/output"
pattern=""

find "$directory" -type f -name "*humaneval*-sanitized.jsonl" -print0 | while IFS= read -r -d '' jsonl_file; do
    if [[ $jsonl_file != *$pattern* ]]; then
        continue
    fi
    echo "  Processing JSONL File: $jsonl_file"
    python evalplus/evaluate.py --dataset humaneval --samples $jsonl_file --i-just-wanna-run
done
find "$directory" -type f -name "*mbpp*-sanitized.jsonl" -print0 | while IFS= read -r -d '' jsonl_file; do
    if [[ $jsonl_file != *$pattern* ]]; then
        continue
    fi
    echo "  Processing JSONL File: $jsonl_file"
    python evalplus/evaluate.py --dataset mbpp --samples $jsonl_file --i-just-wanna-run
done


