"""
第 6 步：Model Routing（按任务难度选模型）—— 在 04 循环前加一个"分流员"

学习目标（对比 04，只多了"先判断难度、再选模型"这一个动作）：
    1. 04 里所有请求都走同一个 MODEL。但"今天天气如何"和"重构这段代码"
       根本不该用同一个模型——前者用便宜小模型就够，后者才值得上贵模型。
    2. 这就是"模型路由"：在调用前，先用一个【便宜模型】判断任务难度，
       再把请求【分发】到便宜模型或昂贵模型。
    3. 为什么值得做：业界统计 ~80% 的调用不需要旗舰模型；
       路由能省 60-95% 成本（代价：多一次判断调用 + 结果不可预测）。

运行（在项目根目录）：
    python steps/06_model_routing.py "列出当前目录的文件"        # 大概率走便宜模型
    python steps/06_model_routing.py "读完README并分析这个项目的设计优缺点"  # 大概率走贵模型

对应你学过的概念：
    - 这是"做法B Routing"的工业版，且用的是其中的 LLM-as-Router（让模型当分流员）
    - 注意：路由的是【选哪个模型】，不是【走哪条业务分支】——别和内容路由搞混

可配置（环境变量）：
    OPENAI_CHEAP_MODEL   便宜模型，默认 gpt-4o-mini
    OPENAI_STRONG_MODEL  昂贵模型，默认 gpt-4o
"""

import json
import os
import sys

from openai import OpenAI


# 两个档次的模型，都可用环境变量覆盖，方便你接 DeepSeek/通义 等兼容接口
CHEAP_MODEL = os.environ.get("OPENAI_CHEAP_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.environ.get("OPENAI_STRONG_MODEL", "gpt-4o")

# ---------------- 和 04 完全相同的部分：工具 ----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出某个目录下的文件和子目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前目录"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的全部内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径"}
                },
                "required": ["path"],
            },
        },
    },
]


def list_dir(path="."):
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(空目录)"
    except Exception as error:
        return f"Error: {error}"


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as error:
        return f"Error: {error}"


FUNCTIONS = {
    "list_dir": list_dir,
    "read_file": read_file,
}


def create_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY，再运行本脚本。")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


# ---------------- 新增 ①：分流员（用一个便宜模型判断难度） ----------------

def classify_complexity(client, task):
    """
    路由决策：让【便宜模型】读一遍任务，只回答 "simple" 或 "complex"。

    设计要点：
    - 判断本身也要花钱，所以一定用【最便宜的模型】来做，否则省的钱又被判断吃回去。
    - 这一步不带工具，纯粹是个"分类器"。
    - 输出做兜底：只要没明确说 complex，就一律当 simple（错判成便宜模型代价小）。
    """
    response = client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是任务难度分流员。判断规则：\n"
                    "- simple：单步、只读、查找、简单总结，凭一次操作就能完成。\n"
                    "- complex：需要多步推理、跨文件综合分析、规划或权衡取舍。\n"
                    "只输出一个词：simple 或 complex，不要任何其他文字。"
                ),
            },
            {"role": "user", "content": task},
        ],
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    return "complex" if "complex" in verdict else "simple"


# ---------------- 新增 ②：核心循环（和 04 几乎一样，只多一个 model 参数） ----------------

def run_agent(client, task, model, max_iterations=8):
    """
    对比 04：唯一的改动是把写死的 MODEL 换成了传入的 model 参数。
    其余"请求模型 → 执行工具 → 回填结果 → 再请求"的循环，和 04 一字不差。
    这说明：路由只是【在循环外面换了个模型名】，循环骨架完全不用动。
    """
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，可以使用工具。"},
        {"role": "user", "content": task},
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'-' * 60}")
        print(f"第 {iteration} 轮（用 {model}）：把 {len(messages)} 条消息发给模型")
        print(f"{'-' * 60}")

        response = client.chat.completions.create(
            model=model,          # ← 这里用的就是分流后选定的模型
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"模型请求：{name}({arguments})")
            if name not in FUNCTIONS:
                result = f"Error: Unknown tool '{name}'"
            else:
                result = FUNCTIONS[name](**arguments)
            print(f"执行结果（前 200 字符）：{str(result)[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "Max iterations reached（达到最大轮数，任务可能没有完成）"


# ---------------- 外层：先分流，再选模型，再跑循环 ----------------

def main():
    client = create_client()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "先看看当前目录里有哪些文件，再读取 README.md，用三句话总结这个项目。"

    print(f"任务：{task}")

    # 1) 分流员判断难度
    complexity = classify_complexity(client, task)

    # 2) 根据难度选模型
    chosen = STRONG_MODEL if complexity == "complex" else CHEAP_MODEL
    print(f"\n{'=' * 60}")
    print(f"【路由决策】任务判定为：{complexity}  →  选用模型：{chosen}")
    print(f"{'=' * 60}")

    # 3) 带着选定的模型跑循环
    answer = run_agent(client, task, model=chosen)
    print(f"\n{'=' * 60}")
    print(f"最终回答：\n{answer}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
