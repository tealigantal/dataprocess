"""
IELTS Part-2 数据生成 & 复制脚本
--------------------------------
依赖:  pip install openai python-dotenv
环境:  .env 里写 DEEPSEEK_API_KEY=sk-xxxx
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ─── OpenAI ≥1.x 客户端 (DeepSeek 兼容) ─────────────────────────
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """单例获取 OpenAI 客户端，指向 DeepSeek API"""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置")
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        )
    return _client


# ─── 工具函数 ─────────────────────────────────────────────────
def load_env() -> None:
    """读取本目录 .env 到环境变量"""
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_sample() -> Dict[str, Any]:
    """取示范 JSON 第一条，保留所需字段"""
    path = Path(__file__).with_name("lora_qwen_finetuning.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    sample = data[next(iter(data))]
    return {
        "plain": sample.get("plain", ""),
        "errors": sample.get("errors", []),
        "output": sample.get("output", {}),
        "output_expanded": sample.get("output_expanded", {}),
    }


def load_topics() -> List[str]:
    path = Path(__file__).with_name("topics.txt")
    return [t.strip() for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]


def build_messages(topic: str, example: Dict[str, Any]) -> List[Dict[str, str]]:
    sys_prompt = (
        "你是雅思口语数据生成助手。请输出与示例结构完全一致的 JSON，仅含 "
        "plain, errors, output, output_expanded 字段，不要 tokens。"
    )
    user_prompt = (
        "示例:\n" + json.dumps(example, ensure_ascii=False, indent=2) +
        f"\n\n话题: {topic}\n请生成新样本。"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 删首行 ``` 或 ```json
        lines = lines[1:]
        # 删末行 ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ─── DeepSeek 调用 ─────────────────────────────────────────────
def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    resp = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ─── 生成 & 校验 ───────────────────────────────────────────────
def generate_one(topic: str, example: Dict[str, Any]) -> Dict[str, Any]:
    content = strip_code_fence(call_deepseek(build_messages(topic, example)))
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}\n返回内容:\n{content}")


def basic_checks(d: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "结果不是 dict"
    if "plain" not in d or "output" not in d:
        return False, "缺少 plain 或 output"
    for i, err in enumerate(d.get("errors", [])):
        word = err.get("word") if isinstance(err, dict) else None
        if word and word not in d["plain"]:
            return False, f"errors[{i}].word 未出现在 plain"
    return True, ""


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


# ─── 批量逻辑 ────────────────────────────────────────────────
def bulk_generate(n: int, example: Dict[str, Any], topics: List[str], out: Path) -> None:
    ok_cnt = 0
    for idx in range(1, n + 1):
        for attempt in range(3):
            try:
                data = generate_one(random.choice(topics), example)
                valid, msg = basic_checks(data)
                if valid:
                    append_jsonl(out, data)
                    ok_cnt += 1
                    print(f"[{idx}/{n}] ✓")
                    break
                print(f"[{idx}/{n}] 校验失败: {msg}")
            except Exception as e:
                print(f"[{idx}/{n}] 生成异常: {e}")
        else:
            print(f"[{idx}/{n}] ⚠ 放弃")
    print(f"批量结束: 成功 {ok_cnt}/{n} 条")


# ─── 交互式入口 ───────────────────────────────────────────────
def interactive_loop() -> None:
    load_env()
    example = load_sample()
    topics = load_topics()
    out_path = Path(__file__).with_name("augmented.jsonl")

    while True:
        try:
            data = generate_one(random.choice(topics), example)
        except Exception as e:
            print(f"[error] {e}\n")
            continue

        print(json.dumps(data, ensure_ascii=False, indent=2))
        cmd = input("y=保存, n=丢弃, 输入“进行数据复制”开始批量: ").strip()

        if cmd == "y":
            ok, msg = basic_checks(data)
            if ok:
                append_jsonl(out_path, data)
                print("✓ 已保存\n")
            else:
                print(f"✗ 校验失败: {msg}\n")

        elif cmd == "进行数据复制":
            try:
                n = int(input("生成数量 N: ").strip())
                bulk_generate(n, example, topics, out_path)
            except ValueError:
                print("N 需为整数\n")

        elif cmd == "n":
            print("已丢弃\n")
        else:
            print("指令未识别\n")


# ─── 主执行 ───────────────────────────────────────────────────
if __name__ == "__main__":
    interactive_loop()
