import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import openai
except ImportError:  # openai may not be installed; placeholder for type check
    openai = None  # type: ignore


def load_env() -> None:
    """Load environment variables from .env if present."""
    env_file = Path(__file__).with_name('.env')
    if env_file.exists():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


def load_sample() -> Dict[str, Any]:
    """Return the first entry of lora_qwen_finetuning.json as example."""
    data_path = Path(__file__).with_name('lora_qwen_finetuning.json')
    with data_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    first_key = next(iter(data))
    example = data[first_key]
    # Only keep fields required in output
    example = {
        'plain': example.get('plain', ''),
        'errors': example.get('errors', []),
        'output': example.get('output', {}),
        'output_expanded': example.get('output_expanded', {}),
    }
    return example


def load_topics() -> List[str]:
    topics_path = Path(__file__).with_name('topics.txt')
    with topics_path.open('r', encoding='utf-8') as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def build_messages(topic: str, example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create prompt messages for DeepSeek."""
    example_json = json.dumps(example, ensure_ascii=False, indent=2)
    system_prompt = (
        "你是雅思口语数据生成助手。请根据给定话题生成与示例结构相同的 JSON，"
        "字段仅包含 plain, errors, output, output_expanded。"
        "不要生成 tokens 字段。"
    )
    user_prompt = f"示例:\n{example_json}\n\n话题: {topic}\n请根据该话题生成新的样本。"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    if openai is None:
        raise RuntimeError("openai library is required to call the API")
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise RuntimeError('DEEPSEEK_API_KEY is not set')
    openai.api_key = api_key
    api_base = os.environ.get('DEEPSEEK_API_BASE')
    if api_base:
        openai.api_base = api_base
    response = openai.ChatCompletion.create(
        model="deepseek-chat", messages=messages, temperature=temperature
    )
    return response['choices'][0]['message']['content']


def generate_one(topic: str, example: Dict[str, Any]) -> Dict[str, Any]:
    messages = build_messages(topic, example)
    content = call_deepseek(messages)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {content}")
    return data


def basic_checks(data: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, 'result is not a dict'
    if 'plain' not in data or 'output' not in data:
        return False, 'missing plain or output'
    if 'errors' in data:
        plain = data.get('plain', '')
        for i, err in enumerate(data['errors']):
            word = None
            if isinstance(err, dict):
                word = err.get('word') or err.get('w')
            if word and word not in plain:
                return False, f"errors[{i}].word not in plain"
    return True, ''


def append_to_file(path: Path, data: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def interactive_loop():
    load_env()
    example = load_sample()
    topics = load_topics()
    out_path = Path(__file__).with_name('augmented.jsonl')

    while True:
        topic = random.choice(topics)
        try:
            data = generate_one(topic, example)
        except Exception as e:
            print(f"[error] {e}")
            continue
        print(json.dumps(data, ensure_ascii=False, indent=2))
        choice = input('输入 y 保存, n 丢弃, 或输入 “进行数据复制” 批量生成: ').strip()
        if choice == 'y':
            ok, msg = basic_checks(data)
            if ok:
                append_to_file(out_path, data)
                print('已保存\n')
            else:
                print(f'未通过校验: {msg}\n')
        elif choice == '进行数据复制':
            try:
                n = int(input('输入生成条数 N: ').strip())
            except ValueError:
                print('输入无效')
                continue
            bulk_generate(n, example, topics, out_path)
        elif choice == 'n':
            print('已丢弃\n')
        else:
            print('未识别的指令\n')


def bulk_generate(n: int, example: Dict[str, Any], topics: List[str], out_path: Path) -> None:
    count = 0
    for _ in range(n):
        for attempt in range(3):
            topic = random.choice(topics)
            try:
                data = generate_one(topic, example)
            except Exception as e:
                print(f"[retry {attempt+1}] {e}")
                continue
            ok, msg = basic_checks(data)
            if ok:
                append_to_file(out_path, data)
                count += 1
                break
            else:
                print(f"[retry {attempt+1}] basic check failed: {msg}")
        else:
            print("放弃该条")
    print(f"批量生成完成，成功 {count}/{n} 条")


if __name__ == '__main__':
    interactive_loop()
