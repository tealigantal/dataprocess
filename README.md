# DataProcess Repository

This repository contains a small dataset for fine-tuning models and a spreadsheet with scores.

## Files

- `lora_qwen_finetuning.json` - JSON dataset with transcripts and scoring information.
- `score.xlsx` - Spreadsheet with scoring results (binary, not displayed here).

## Usage

A helper script `analyze_dataset.py` is provided to calculate basic statistics from the JSON file.
Run it using:

```bash
python3 analyze_dataset.py
```

This will print the number of entries, the average token count, and the average overall score.
