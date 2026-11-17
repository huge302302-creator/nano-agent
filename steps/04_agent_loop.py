"""
第 4 步：完整的 Agent 循环（素版）

学习目标：
    1. 把“请求模型 → 执行工具 → 回填结果 → 再请求”放进一个循环
    2. 循环的退出条件：模型不再返回 tool_calls
    3. max_iterations 的作用：防止模型停不下来，陷入无限调用

运行（在项目根目录）：
    python steps/04_agent_loop.py
    python steps/04_agent_loop.py "看看当前目录有哪些文件，再读一下 README.md"

为了安全，这个素版只提供两个“只读”工具：
    read_file  读文件
    list_dir   列目录

根目录的 agent.py 就是在这个骨架上，增加了 write_file / execute_bash、
人工确认和错误处理。先看懂这里的循环，再去看 agent.py，会轻松很多。
"""

import json
import os
import sys

from openai import OpenAI


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# 给模型看的工具菜单
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


# ---------------- 工具实现 ----------------

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


# 名字 → 函数
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


# ---------------- 核心循环 ----------------

def run_agent(client, task, max_iterations=8):
    """整个 Agent 只有这一个循环，和根目录 agent.py 的结构完全一致。"""
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，可以使用工具。"},
        {"role": "user", "content": task},
    ]            ##这是发给大模型的messages，里面有用户的任务和之前的对话记录
    #role: "system"：给模型的人设/规则说明书，"你是一个简洁的助手，可以使用工具"
    #role: "user"：用户的任务，比如"看看目录再读 README.md"

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'-' * 60}")
        print(f"第 {iteration} 轮：把 {len(messages)} 条消息发给模型")
        print(f"{'-' * 60}")
        #只是帮你观察 Agent 每轮在干什么，学习阶段很有用，并不是关键部分

        response = client.chat.completions.create(   #client是前面建好的客户端
            model=MODEL,          #用哪个模型
            messages=messages,         #对话历史
            tools=TOOLS,         #工具菜单
        )
        message = response.choices[0].message        #取出模型的回复
        messages.append(message)                  #把模型的回复也append进对话历史中

        # 没有 tool_calls：模型给出了最终回答，循环结束
        if not message.tool_calls:
            return message.content

        # 有 tool_calls：逐个执行
        for tool_call in message.tool_calls:      #循环：模型可能一次要求调用多个工具
            name = tool_call.function.name
            # 素版没有做 JSON 解析保护；生产版 agent.py 会处理坏参数。
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
        #执行工具进入messages，下一轮循环会把这个结果发给模型，让模型继续思考下一步要做什么
        # 本轮工具执行完，进入下一轮

    return "Max iterations reached（达到最大轮数，任务可能没有完成）"


def main():                   #程序的主调度
    client = create_client()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])   #如果跟着有命令的话，把命令取出来加入task
    else:
        task = "先看看当前目录里有哪些文件，再读取 README.md，用三句话总结这个项目。"  #如果不跟着命令，则会按照这个默认去走
    ## 敲法一：只敲脚本名，后面什么都不加
     #python steps/04_agent_loop.py
    ## 敲法二：脚本名后面跟一句话
     #python steps/04_agent_loop.py "帮我看看目录里有什么"
    print(f"任务：{task}")
    answer = run_agent(client, task)
    print(f"\n{'=' * 60}")
    print(f"最终回答：\n{answer}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
