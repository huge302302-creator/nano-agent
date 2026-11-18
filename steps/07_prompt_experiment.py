"""
第 7 步：提示词实验——同一个任务，只换 system 提示词，看答案变多少

实验目的（对应 hello-agents 3.2.1 Prompt Engineering 和第九章 9.2.1）：
    1. 亲眼验证："模型的能力上限取决于它看见了什么"
       —— 同一个模型、同一个问题，只改 system 那一行，输出天差地别
    2. 练习提示词四要素：任务 / 背景 / 限制 / 输出要求
    3. 这个脚本【不带任何工具】——因为要隔离变量：这次只研究"提示词"这一个因素

实验设计：
    A 组（基线）  ：一句普通 system——你现在 04 里用的就是这种
    B 组（四要素）：背景（教小白）+ 限制（禁术语/不超3句）+ 输出要求（每句以"想象"开头）
    C 组（反向人设）：资深架构师——专业、精确、必须用标准术语
    同一个问题问三遍，把三份回答并排打印，你自己对比。

运行（在项目根目录）：
    python steps/07_prompt_experiment.py
    python steps/07_prompt_experiment.py "什么是上下文窗口？"
"""

import os
import sys

from openai import OpenAI


# 模型名做成可配置：设 DEMO_MODEL 环境变量可覆盖；默认用 DeepSeek Pro 档（结果对比更清晰）
MODEL = os.environ.get("DEMO_MODEL", "deepseek-v4-pro")


def create_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY，再运行本脚本。")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


# ---------------- 三组实验材料：唯一的变量是 system 这一行 ----------------

EXPERIMENTS = [
    (
        "A 组（基线，你现在 04 用的就是这种）",
        "你是一个简洁的助手。",
    ),
    (
        "B 组（四要素齐全：背景+限制+输出要求）",
        (
            "你是一位专门教完全零基础人群的 Python 老师（背景）。"
            "解释时只准用小学生词汇和生活类比，禁止出现任何英文术语（限制）。"
            "总共不超过 3 句话，且每句话必须以'想象'两个字开头（输出要求）。"
        ),
    ),
    (
        "C 组（反向人设：资深架构师）",
        "你是一位有 15 年经验的资深软件架构师。回答必须专业、精确、严谨，"
        "必须使用标准技术术语，并给出严格定义。",
    ),
]


def ask(client, system_prompt, question):
    """不带工具的纯对话调用——和 01_hello_llm.py 同款，只是 system 可换。"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def main():
    client = create_client()

    # 问题可以从命令行传，方便你换着做实验
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "什么是 Agent 循环？"

    print(f"实验问题：{question}\n")

    for title, system_prompt in EXPERIMENTS:
        print("=" * 60)
        print(f"【{title}】")
        print(f"system 提示词：{system_prompt}")
        print("-" * 60)
        answer = ask(client, system_prompt, question)
        print(answer)
        print()

    print("=" * 60)
    print("实验作业：")
    print("  1. 对比三组：谁的回答最短？谁用了术语？谁做到了'每句以想象开头'？")
    print("  2. 自己在 EXPERIMENTS 里加一个 D 组提示词，再跑一次，观察变化")
    print("  3. 换个问题再跑：python steps/07_prompt_experiment.py \"什么是RAG？\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
