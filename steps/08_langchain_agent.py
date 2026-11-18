"""
第 8 步：用 LangChain 重写 04 —— 同一任务、同一工具，看看框架接管了什么

学习目标：
    1. 亲手验证"框架 = 把你手写过的东西封装成零件"
    2. 对照表（04 手写的 vs LangChain 接管的）：
       ┌────────────────────────────┬──────────────────────────────────┐
       │ 04 里你手写的               │ 在 LangChain 里变成了             │
       ├────────────────────────────┼──────────────────────────────────┤
       │ TOOLS（手写 JSON schema）   │ @tool 装饰器：从函数签名+docstring │
       │ FUNCTIONS（名字→函数注册表） │ 自动生成并注册，两份合一           │
       │ create_client() + MODEL    │ init_chat_model(...)              │
       │ run_agent 整个 for 循环     │ create_agent(...) 内部就是它      │
       │ 请求→tool_calls→执行→回填   │ 框架自动完成                      │
       │ max_iterations 保险丝       │ 框架内部处理                      │
       │ messages.append(...) 手工   │ 框架替你维护历史                  │
       │ 你要写的只剩：工具函数 + system_prompt + 调用                    │
       └────────────────────────────┴──────────────────────────────────┘
    3. 脚本最后的"解剖黑盒"部分：把框架内部维护的完整消息历史打印出来，
       你会看到 HumanMessage → AIMessage(tool_calls) → ToolMessage → ...
       ——这不就是你 04 里那个不断变长的 messages 列表吗？黑盒就此透明。

安装（第一次用框架，需要先装包）：
    pip install -U langchain langchain-openai

运行（在项目根目录，OPENAI_API_KEY 环境变量和之前一样，不用改）：
    python steps/08_langchain_agent.py
    python steps/08_langchain_agent.py "看看当前目录有哪些文件，再读一下 README.md"
"""

import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# ---------------- 工具：04 的 TOOLS + FUNCTIONS，现在一个 @tool 搞定 ----------------
#
# 对比 04：那里你写了两遍——TOOLS 里手写 JSON 描述"叫什么、参数是什么"，
# FUNCTIONS 字典里再把名字映射到函数。
# 现在 @tool 装饰器直接读【函数名 + 参数类型 + docstring】自动生成同样的 JSON schema
# 并注册。docstring 会原样发给模型看——所以描述要写清楚，和 04 的结论一致。


@tool
def list_dir(path: str = ".") -> str:
    """列出某个目录下的文件和子目录。path 是目录路径，默认当前目录。"""
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(空目录)"
    except Exception as error:
        return f"Error: {error}"


@tool
def read_file(path: str) -> str:
    """读取一个文本文件的全部内容。path 是要读取的文件路径。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as error:
        return f"Error: {error}"


# ---------------- 模型：04 的 create_client() + MODEL 常量 ----------------
#
# init_chat_model 会自动读取 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量，
# 和你 .env 里已有的配置完全一致，什么都不用改。
# 想换提供商/模型，只改这一个字符串。

model = init_chat_model(
    os.environ.get("DEMO_MODEL", "deepseek-v4-pro"),  # 默认 Pro 档；设 DEMO_MODEL 可覆盖
    model_provider="openai",
)


# ---------------- 组装：04 的整个 run_agent 函数，浓缩成这一句 ----------------

agent = create_agent(
    model=model,
    tools=[list_dir, read_file],
    system_prompt="你是一个简洁的助手，可以使用工具。",  # 和 04 一字不差，方便对照
)


def dissect(result_messages):
    """
    解剖黑盒：把框架替你维护的完整消息历史逐条打印出来。

    你会看到 LangChain 的消息类型，和 04 里的 role 一一对应：
        HumanMessage    = role: "user"
        AIMessage       = role: "assistant"（可能带 tool_calls）
        ToolMessage     = role: "tool"（带 tool_call_id，和请求配对）
        SystemMessage   = role: "system"
    """
    print("\n" + "=" * 60)
    print("【解剖黑盒】框架内部替你维护的完整消息历史：")
    print("=" * 60)
    for i, m in enumerate(result_messages):
        kind = type(m).__name__
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            names = [f"{tc['name']}({tc['args']})" for tc in tool_calls]
            print(f"  [{i}] {kind:<14} → 请求调用工具: {names}")
        elif kind == "ToolMessage":
            preview = str(m.content)[:60].replace("\n", "\\n")
            print(f"  [{i}] {kind:<14} → 工具结果(前60字符): {preview}")
        else:
            preview = str(m.content)[:60].replace("\n", "\\n")
            print(f"  [{i}] {kind:<14} → {preview}")
    print(f"\n历史总长度：{len(result_messages)} 条消息"
          f"（04 里这个列表是你手工 append 的，这里框架替你 append 了）")


def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "先看看当前目录里有哪些文件，再读取 README.md，用三句话总结这个项目。"

    print(f"任务：{task}")

    # 对比 04：run_agent(client, task) 里整个循环干的活，
    # 现在一次 invoke 调用，框架内部自己转循环直到模型不再请求工具。
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    dissect(result["messages"])

    print(f"\n{'=' * 60}")
    print("最终回答：")
    print(result["messages"][-1].content)   # 历史里最后一条 AIMessage = 最终答案
    print("=" * 60)


if __name__ == "__main__":
    main()
