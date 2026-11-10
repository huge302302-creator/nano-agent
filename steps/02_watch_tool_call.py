"""
第 2 步：让模型“看见”工具，但先不执行

学习目标：
    1. 理解 tools 参数是一份给模型看的“工具菜单”（JSON schema）
    2. 亲眼看到模型收到菜单后，返回的不是文本，而是 tool_calls
    3. 记住：模型只会“提出请求”，永远不会自己执行工具

运行（在项目根目录）：
    python steps/02_watch_tool_call.py

本脚本故意不执行任何工具。运行后你会发现 README.md 根本没有被读取。
"""

import json
import os

from openai import OpenAI


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# 工具说明书：告诉模型“有一个 read_file 工具，它需要一个 path 参数”。
# 这份 JSON 不包含任何实现代码，只是元数据。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文本文件的全部内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径",
                    }
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


def main():
    client = create_client()

    messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "请读取 README.md，然后告诉我这个项目是做什么的。"},
    ]

    print("＝" * 60)
    print("第 1 次请求模型（这次带上 tools）…")
    print("＝" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,  # 关键：把工具菜单一起发给模型
    )

    message = response.choices[0].message

    # 模型如果决定使用工具，content 通常为空，tool_calls 不为空。
    if not message.tool_calls:
        print("模型这次没有请求工具，直接回答了：")
        print(message.content)
        print("\n提示：可以把任务写得更明确，例如“必须使用 read_file 读取 README.md”。")
        return

    print("模型没有直接回答，而是提出了下面这些工具调用请求：\n")
    for tool_call in message.tool_calls:
        print(f"  tool_call.id : {tool_call.id}")
        print(f"  工具名        : {tool_call.function.name}")

        # arguments 是一段 JSON 字符串，不是 dict，需要自己解析。
        print(f"  原始参数      : {tool_call.function.arguments}")
        print(f"  解析后的参数  : {json.loads(tool_call.function.arguments)}")
        print()

    print("＝" * 60)
    print("请确认：README.md 被读取了吗？没有。")
    print("模型只是说“我想调用 read_file”，执行权在我们手里。")
    print("下一步（03）我们就真正执行它，并把结果交还给模型。")
    print("＝" * 60)


if __name__ == "__main__":
    main()
