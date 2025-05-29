import json
import argparse
from pathlib import Path

def convert_json_to_jsonl(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of JSON objects.")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + "\n")

    print(f"Converted {len(data)} records from {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON array file to JSONL format.")
    parser.add_argument("input", type=Path, help="Path to the input .json file")
    parser.add_argument("output", type=Path, help="Path to the output .jsonl file")

    args = parser.parse_args()
    convert_json_to_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
