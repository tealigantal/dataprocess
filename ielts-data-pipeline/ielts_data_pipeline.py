import json, random, re, time, os, textwrap
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

"""IELTS Part‑2 Synthetic Data Generator (DeepSeek)
--------------------------------------------------
• 运行脚本直接进入审核循环。
  - Enter/空行   : 生成 1 条候选 → Y 保存 / n 丢弃
  - x            : 批量模式 (输入数量后自动保存)
  - q            : 退出
• Debug：格式不符的原始返回写入 `bad_raw.log`。
  (修正 2025‑06‑23：不再使用 Path.write_text(append=True)，改为标准文件追加)
"""

# ------------------ 0. DeepSeek 客户端 ------------------ #
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

RAW_LOG = Path("bad_raw.log")
TRAIN_FILE = Path("train.jsonl")

# ------------------ 0.b 服务器可用性检查 ------------------ #

def check_deepseek(timeout: int = 10) -> bool:
    """简易连通性测试：发送最小请求并捕获异常。"""
    try:
        _ = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
            top_p=0.1,
            timeout=timeout,
        )
        return True
    except Exception as err:  # 广泛捕获，避免依赖 openai 内部异常类型
        print(f"❌ DeepSeek API 不可用: {err}")
        return False

# ------------------ 1. Few‑shot 样本 ------------------ #

def load_examples(k: int = 3) -> List[Dict[str, Any]]:
    p = Path(__file__).with_name("lora_qwen_finetuning.json")
    data = list(json.loads(p.read_text(encoding="utf-8")).values())
    random.shuffle(data)
    return [
        {
            "plain": d["input"]["plain"],
            "pause_count": d["input"]["pause_count"],
            "pause_total": d["input"]["pause_total"],
            "errors": d["input"].get("errors", []),
            "score": d["output"]["score"],
            "explanation": d["output"]["explanation"],
        }
        for d in data[:k]
    ]

# ------------------ 2. Topic 列表 ------------------ #

def load_topics() -> List[str]:
    return [
        t.strip()
        for t in Path(__file__).with_name("topics.txt").read_text(encoding="utf-8").splitlines()
        if t.strip()
    ]

# ------------------ 3. Prompt ------------------ #
SCHEMA = (
    "{"  # 紧凑写法
    '"input":{"plain":str,"pause_count":int,"pause_total":float,"errors":list[str]},'
    '"output":{"score":str,"explanation":str}'
    "}"
)

BASE_SYS = (
    "你是 IELTS Part‑2 样本生成助手。只返回一行 JSON (不加 markdown)。\n"
    f"结构必须为: {SCHEMA}\n"
    "文本 190‑230 词, ≥2 C1 形容词, ≥3 连接词, ≥3 复杂句, 停顿统计匹配长度。"
    "判断 errors 字段时，需忽略大小写、连字符与标点等形式差异，仅记录真正影响含义的拼写/语法错误。"
)

# ------------------ 4. 验证 & 清理 ------------------ #

def _rand_pause(words: int) -> tuple[int, float]:
    pc = max(1, int(words * random.uniform(0.03, 0.07)))
    return pc, round(pc * random.uniform(0.6, 1.2), 2)


def _clean_raw(raw: str) -> str:
    """去掉 ```json ...``` 或反引号围栏"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return raw


def validate(raw: str) -> Dict[str, Any] | None:
    raw = _clean_raw(raw)
    try:
        d = json.loads(raw)
    except Exception:
        return None
    need_in = {"plain", "pause_count", "pause_total", "errors"}
    need_out = {"score", "explanation"}
    if (
        isinstance(d, dict)
        and "input" in d
        and "output" in d
        and set(d["input"]) >= need_in
        and set(d["output"]) >= need_out
    ):
        return d
    return None

# ------------------ 5. 生成 ------------------ #

def build_messages(topic: str, exs: List[Dict[str, Any]]):
    band = random.choice(["5.0-5.5", "6.0-6.5", "7.0-7.5"])
    few = "\n\n".join(
        [f"示例{i+1}:\n{json.dumps(e,ensure_ascii=False,indent=2)}" for i, e in enumerate(exs)]
    )
    user = f"{few}\n\n话题: {topic}\n目标总分: {band}\n生成全新样本。"
    return [
        {"role": "system", "content": BASE_SYS},
        {"role": "user", "content": user},
    ]


def generate_one(topic: str, exs: List[Dict[str, Any]]):
    msgs = build_messages(topic, exs)
    for attempt in range(3):
        try:
            rsp = client.chat.completions.create(
                model="deepseek-chat",
                messages=msgs,
                temperature=1.0,
                top_p=0.9,
            ).choices[0].message.content
        except Exception as api_err:
            raise RuntimeError(f"API 调用失败: {api_err}")
        data = validate(rsp)
        if data:
            if not data["input"]["pause_count"] or not data["input"]["pause_total"]:
                w = len(re.findall(r"\\b\\w+\\b", data["input"]["plain"]))
                data["input"]["pause_count"], data["input"]["pause_total"] = _rand_pause(w)
            return data
        # 写入日志（追加）
        with RAW_LOG.open("a", encoding="utf-8") as log:
            log.write(
                textwrap.dedent(
                    f"\n===== Attempt {attempt+1} =====\n{rsp[:1000]}\n"
                )
            )
        print("⚠️ 模型返回格式不符，已写入 bad_raw.log (首 200 字):", rsp[:200])
        time.sleep(1)
    raise RuntimeError("多次生成均失败，检查提示词/网络")

# ------------------ 6. 保存 ------------------ #

def save(obj: Dict[str, Any]):
    with TRAIN_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ------------------ 7. 主循环 ------------------ #

def main():
     # 启动前检查 DeepSeek 服务器状态
    if not check_deepseek():
        print("请检查网络或 API Key 后重试。程序已退出。")
        return
    topics = load_topics()
    examples = load_examples(k=5)
    saved = 0
    print("进入审核模式: Y 保存 / n 丢弃 / x 批量复制 / q 退出\n")
    while True:
        cmd = input("指令(Enter=生成): ").strip().lower()
        if cmd in {"q", "quit"}:
            print(f"退出。共保存 {saved} 条。"); break
        if cmd == "x":
            try:
                n = int(input("批量数量 > "))
            except ValueError:
                print("数量无效\n"); continue
            for i in range(n):
                topic = random.choice(topics)
                try:
                    samp = generate_one(topic, examples)
                    save(samp); saved += 1
                    print(f"✔ 批量 {i+1}/{n}")
                except Exception as e:
                    print("✖ 失败:", e)
            continue
        # 单条模式
        topic = random.choice(topics)
        try:
            samp = generate_one(topic, examples)
        except Exception as e:
            print("生成失败:", e); continue
        print("\n===== 候选样本 =====")
        print(json.dumps(samp, ensure_ascii=False, indent=2))
        if input("保存? (Y/n) > ").strip().lower() in {"", "y", "yes"}:
            save(samp); saved += 1; print(f"✔ 保存，总计 {saved}\n")
        else:
            print("△ 丢弃\n")

if __name__ == "__main__":
    main()
