cd evaluation/evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)
directory="../evaluate/output"

find "$directory" -type d -print0 | while IFS= read -r -d '' folder; do
    if [[ $folder != ** ]]; then
        continue
    fi
    echo "Processing Folder: $folder"

    find "$folder" -type f -name "*humaneval*.jsonl" ! -name "*-sanitized.jsonl" -print0 | while IFS= read -r -d '' jsonl_file; do
        echo "  Processing JSONL File: $jsonl_file"
        python tools/sanitize.py --samples $jsonl_file --dataset humaneval
    done
    find "$folder" -type f -name "*mbpp*.jsonl" ! -name "*-sanitized.jsonl" -print0 | while IFS= read -r -d '' jsonl_file; do
        echo "  Processing JSONL File: $jsonl_file"
        python tools/sanitize.py --samples $jsonl_file --dataset mbpp
    done
done
