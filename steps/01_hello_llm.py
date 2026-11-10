"""
第 1 步：只和模型聊天，不涉及任何工具

学习目标：
    1. 看清 messages 的结构：system / user / assistant 三种角色
    2. 确认模型本身只会“说话”，不会读文件、不会执行命令

运行（在项目根目录）：
    python steps/01_hello_llm.py

需要先设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL、OPENAI_MODEL）。
"""

import os

from openai import OpenAI


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def create_client():
    """创建客户端，并在缺少 Key 时给出中文提示，而不是抛出一大段 traceback。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 OPENAI_API_KEY，再运行本脚本。")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


def main():
    client = create_client()

    # messages 就是一次对话的全部内容。
    # 模型没有记忆：每次请求都要把历史消息完整带上，它才知道上下文。
    messages = [
        # system：给模型的角色设定和行为约束
        {"role": "system", "content": "你是一个简洁的助手，回答不超过三句话。"},
        # user：用户说的话
        {"role": "user", "content": "用一句话解释：什么是 AI Agent？"},
    ]

    print("＝" * 60)
    print("发给模型的消息：")
    for message in messages:
        print(f"  [{message['role']}] {message['content']}")
    print("＝" * 60)

    # 调用模型。注意这一次没有传 tools，模型只能生成文本。
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    # 回复藏在 response.choices[0].message 里。
    # message.role 是 "assistant"，message.content 是文本内容。
    message = response.choices[0].message
    print("\n模型回复：")
    print(message.content)

    print("\n观察：到这里为止，模型只是“说”了一句话，没有执行任何操作。")
    print("下一步（02）我们会告诉模型有哪些工具可用，看它会怎么回答。")


if __name__ == "__main__":
    main()
