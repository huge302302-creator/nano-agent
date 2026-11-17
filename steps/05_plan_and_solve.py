"""
第 5 步：Plan-and-Solve（先规划，再执行）—— 在 04 循环上加一层"计划"

学习目标（对比 04，只多了"规划"这一个动作）：
    1. 04 的 ReAct 是"走一步看一步"，没有全局计划 → 长任务容易跑偏、忘记目标
    2. Plan-and-Solve 的做法：先逼模型一次性输出一份【步骤清单】，
       再"逐条"执行这个清单。有了清单钉住方向，模型就不会乱跑。
    3. 体会"规划者"和"执行者"的分工：
       - create_plan()  = 只负责想清楚"分几步"（不调用任何工具）
       - run_agent_step() = 负责把某一步真正干完（内部就是你熟悉的 04 循环）

运行（在项目根目录）：
    python steps/05_plan_and_solve.py
    python steps/05_plan_and_solve.py "看看目录，读 README，再用三句话总结这个项目"

和 04 的关系：
    工具菜单 TOOLS、工具实现 FUNCTIONS、create_client() 都和 04 一模一样。
    唯一新增的是"先列计划、再按计划循环执行"这套外层结构。
    这套外层结构，正是 nano-agent 根目录 agent-claudecode.py 里 plan 工具的简化版。
"""

import json
import os
import re
import sys

from openai import OpenAI


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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


# ---------------- 新增 ①：规划阶段（只决策，不动手） ----------------

def _parse_plan(raw_text):
    """
    把模型返回的"计划文本"解析成 Python 列表。

    模型有时会规规矩矩返回 ["步骤1", "步骤2"]，
    有时会自作主张包一层 markdown 代码块，例如：
        ```json
        ["步骤1", "步骤2"]
        ```
    这里先剥掉代码块标记，再用正则抓出第一个 [...] 数组，尽量健壮。
    解析失败就退化成"单步计划"——把整个任务当成一步，保证程序不崩。
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and data:
                return [str(s) for s in data]
        except json.JSONDecodeError:
            pass
    # 兜底：解析不出列表，就退回到"整个任务一步做完"
    return [raw_text.strip()]


def create_plan(client, task):
    """
    规划阶段：单独调用一次模型，**不带任何工具**，让它只输出"分几步"。

    关键设计：
    - 这一步不执行工具。模型只负责"想计划"，不能"动手"。
    - 用 response_format 不是必须，这里靠提示词约束 + 解析，兼容性更好。
    """
    print("\n" + "=" * 60)
    print("【规划阶段】让模型先列出执行计划（此步不调用工具）")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是任务规划助手。请把用户的任务拆解成具体、可执行的步骤。"
                    "只输出一个 JSON 数组，每个元素是一句话的步骤描述。"
                    "不要执行任何操作，不要解释，不要输出多余文字。"
                ),
            },
            {"role": "user", "content": task},
        ],
    )
    plan = _parse_plan(response.choices[0].message.content)

    for i, step in enumerate(plan, 1):
        print(f"  第 {i} 步：{step}")
    return plan


# ---------------- 新增 ②：执行单个步骤（内部就是你熟悉的 04 循环） ----------------

def run_agent_step(client, messages, step_instruction, max_iterations=4):
    """
    执行阶段的最小单元：把"某一步"交给模型，跑一个 04 式的工具循环。

    参数 messages 是【共享的对话历史】——所有步骤共用同一条历史，
    所以第 2 步能看到第 1 步做了什么、结果如何（连续性）。
    副作用：messages 会随着执行不断变长（这就是"全量重发"的雪球）。

    返回：(这一步的最终文字结果, 更新后的 messages)
    """
    # 把当前步骤作为一条 user 消息加进历史
    messages.append({"role": "user", "content": f"现在请执行这一步：{step_instruction}"})

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message)

        # 这一步没再要求调用工具 → 说明这一步做完了，返回它的文字结论
        if not message.tool_calls:
            return message.content or "", messages

        # 有 tool_calls：逐个执行（这段和 04 完全一致）
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"    [工具] {name}({arguments})")
            if name not in FUNCTIONS:
                result = f"Error: Unknown tool '{name}'"
            else:
                result = FUNCTIONS[name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "（这一步达到最大轮数，可能没做完）", messages


# ---------------- 外层总调度：规划 → 逐步执行 → 汇总 ----------------

def run_plan_and_solve(client, task):
    # 1) 共享历史，从"原始任务"开始
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，可以先规划再使用工具执行。"},
        {"role": "user", "content": task},
    ]

    # 2) 规划阶段
    plan = create_plan(client, task)

    # 把计划写进历史，让后续每一步都知道"全局有几步、现在在第几步"
    plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
    messages.append({"role": "system", "content": f"整体执行计划如下：\n{plan_text}"})

    # 3) 执行阶段：逐条执行计划（这就是 Plan-and-Solve 的灵魂）
    results = []
    for i, step in enumerate(plan, 1):
        print(f"\n{'-' * 60}")
        print(f"【执行第 {i}/{len(plan)} 步】{step}")
        print(f"{'-' * 60}")
        result, messages = run_agent_step(client, messages, step)
        results.append(f"第 {i} 步「{step}」的结果：\n{result}")

    # 4) 汇总阶段：把所有步骤结果交给模型，生成对"原始任务"的最终回答
    print("\n" + "=" * 60)
    print("【汇总阶段】合并各步结果，生成最终回答")
    print("=" * 60)
    messages.append({"role": "user", "content": "请基于以上各步结果，给出对原始任务的最终总结回答。"})
    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


def main():
    client = create_client()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "先看看当前目录里有哪些文件，再读取 README.md，用三句话总结这个项目。"

    print(f"任务：{task}")
    answer = run_plan_and_solve(client, task)
    print(f"\n{'=' * 60}")
    print(f"最终回答：\n{answer}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
