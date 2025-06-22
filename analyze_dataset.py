import json
import statistics

DATA_FILE = 'lora_qwen_finetuning.json'

def parse_score(score_str):
    """Return the mid value of a range like '5.0-5.5'."""
    if isinstance(score_str, str) and '-' in score_str:
        start, end = score_str.split('-')
        try:
            return (float(start) + float(end)) / 2
        except ValueError:
            pass
    try:
        return float(score_str)
    except (TypeError, ValueError):
        return None


def main():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokens = []
    overall_scores = []

    for entry in data.values():
        tokens.append(entry.get('tokens', 0))
        score = parse_score(entry.get('output', {}).get('overall'))
        if score is not None:
            overall_scores.append(score)

    print(f'Total entries: {len(data)}')
    if tokens:
        print(f'Average tokens: {statistics.mean(tokens):.2f}')
    if overall_scores:
        print(f'Average overall score: {statistics.mean(overall_scores):.2f}')

if __name__ == '__main__':
    main()
