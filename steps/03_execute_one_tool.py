"""
第 3 步：执行一次工具，并把结果交还给模型（完整的一轮）

学习目标：
    1. 模型提出 tool_calls 后，由我们的 Python 代码真正执行
    2. 结果必须以 role="tool" 的格式追加进 messages
    3. tool_call_id 必须和模型请求里的 id 对应，模型才知道结果属于哪次调用
    4. 把结果再发给模型，它才能基于文件内容回答

运行（在项目根目录）：
    python steps/03_execute_one_tool.py

这一轮结束后，模型才第一次“看见”README.md 的内容。
"""

import json
import os

from openai import OpenAI


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

TOOLS = [
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
    }
]


def create_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY，再运行本脚本。")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


def read_file(path):
    """真正的工具实现：一个普通 Python 函数，参数名和 schema 保持一致。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    client = create_client()

    messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "请读取 README.md，然后用三句话总结这个项目。"},
    ]

    # ---------- 第 1 次请求：模型提出工具调用 ----------
    print("＝" * 60)
    print("第 1 次请求模型：让它决定要不要用工具")
    print("＝" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        print("模型没有使用工具，直接回答：")
        print(message.content)
        return

    # 这条 assistant 消息里带着 tool_calls，必须原样存进历史。
    # 下一轮请求如果缺了它，模型会认为工具调用“没有发生过”。
    messages.append(message)

    # ---------- 执行工具，并回填结果 ----------
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"\n模型请求调用：{name}({arguments})")
        print("这个请求本身不会执行任何东西，下面这行才是真正读文件。")

        if name == "read_file":
            result = read_file(**arguments)
        else:
            result = f"Error: Unknown tool '{name}'"

        print(f"工具执行结果（前 200 个字符）：\n{result[:200]}\n")

        # 关键格式：role="tool"，并带上对应的 tool_call_id。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # ---------- 第 2 次请求：模型基于工具结果回答 ----------
    print("＝" * 60)
    print("第 2 次请求模型：这次它能看到 README.md 的内容了")
    print("＝" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    final_message = response.choices[0].message

    print("\n最终回答：")
    print(final_message.content)

    print("\n观察：这就是一次完整的工具调用。")
    print("如果任务需要多步，只要把“请求→执行→回填→再请求”放进循环，就是 Agent。")
    print("下一步（04）我们加上循环，让它自己决定要调用几次工具。")


if __name__ == "__main__":
    main()
