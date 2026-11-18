"""
第 9 步：LangChain 完整 Agent（带工具调用 + 记忆）—— 参考官方"真实场景智能体"结构

这是学习路线第 8 节的毕业作品：独立写出一个带工具调用的 Agent。
文档把"真实场景"拆成 5 个模块，本文件严格照这个顺序组织，每个模块都标了【作用】
和【对应你学过的东西】，方便你逐块理解：

    模块 1  配置模型     init_chat_model(...)        ← 04 的 create_client()+MODEL
    模块 2  定义工具     @tool × 3                    ← 04 的 TOOLS+FUNCTIONS
    模块 3  系统提示词   SYSTEM_PROMPT                ← 04 的 role:"system"
    模块 4  记忆         InMemorySaver + thread_id    ← 你手写的"存→取→塞"自动化版
    模块 5  组装         create_agent(...)            ← 04 的 run_agent 函数(循环)
    模块 6  运行         agent.invoke(...)            ← 04 的 run_agent(...) 调用

运行（在项目根目录，先 chcp 65001 防中文乱码）：
    pip install -U langchain langchain-openai   （你已装过，可跳过）
    python steps/09_langchain_full_agent.py

本文件会连续做【两轮】对话，第二轮故意问第一轮的内容 ——
让你亲眼看到"记忆(checkpointer)"真的生效，也看到它背后仍是"全量重发"。
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 模块 1：配置模型
# 【作用】建立和大模型的连接，并设定"怎么说话"的参数。
# 【对应】04 的 create_client() + MODEL 常量；这里合成一句。
# 【新增】temperature/max_tokens/timeout —— 04 里没设，靠默认；
#         生产项目要显式控制：温度低=更稳定，max_tokens=防超长回答，timeout=防卡死。
# ============================================================
model = init_chat_model(
    os.environ.get("DEMO_MODEL", "deepseek-v4-pro"),  # 默认 Pro 档；设 DEMO_MODEL 可覆盖
    model_provider="openai",   # 自动读 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量
    temperature=0.2,           # 0~1，越小越稳定、越少发挥（工具型 Agent 常调低）
    max_tokens=2000,           # 单次回答最多生成多少 token，防止失控刷字
    timeout=60,                # 网络超时，防止请求无限挂起
)


# ============================================================
# 模块 2：定义工具
# 【作用】给 Agent 提供"手脚"——它能调用的外部能力。
# 【对应】04 的 TOOLS(手写JSON) + FUNCTIONS(名字→函数)，两份合一。
# 【关键】@tool 只帮你把"函数名+参数类型+docstring"自动转成 JSON schema；
#         工具真正干什么活，全在函数体里，是你写的（上次你已经想通了这点）。
# ============================================================

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


@tool
def calculator(expression: str) -> str:
    """计算一个算术表达式，支持 + - * / 和括号。expression 是算式字符串，例如 "3*4+10"。"""
    # 安全起见：只允许数字、运算符、空格和括号，杜绝任意代码执行。
    # （生产级更稳妥的做法是用 ast.literal_eval 或专用解析器，这里为教学保持简单。）
    if not all(c in "0123456789+-*/(). " for c in expression):
        return "Error: 表达式含非法字符，只支持数字和 + - * / ( ) 。"
    try:
        return str(eval(expression))   # 已通过上面的字符白名单过滤
    except Exception as error:
        return f"Error: {error}"


# 把三个工具装进列表，交给 create_agent（对应 04 的 FUNCTIONS 字典）
tools = [list_dir, read_file, calculator]


# ============================================================
# 模块 3：系统提示词
# 【作用】定义 Agent 的"人设 + 规则"，每轮都在上下文里（第①样成分）。
# 【对应】04 的 {"role":"system",...}；这里写得更结构化（文档推荐做法）。
# 【关键】文档强调：系统提示词要"具体且可执行"，把能力边界写清楚，
#         能显著减少模型乱调工具。这就是你学过的"提示词四要素"。
# ============================================================
SYSTEM_PROMPT = """你是一个严谨的文件研究助手。

## 能力
- list_dir：列出目录内容
- read_file：读取文件
- calculator：做算术

## 规则
- 需要文件信息时，先用工具读取，不要凭空猜测内容。
- 计算类问题必须调用 calculator，不要自己心算（你会算错）。
- 回答用中文，简洁。"""


# ============================================================
# 模块 4：记忆（checkpointer）
# 【作用】把"对话历史"按 thread_id 自动存下来，下次同 thread 自动接上。
# 【对应】你手写的 load_memory/save_memory "存→取→塞"，这里框架自动化。
# 【注意】InMemorySaver 存内存，重启即失忆；生产可换成 Postgres/Redis 版，
#         接口不变。它解决"记得住"，不解决"记得省"（历史仍会全量重发、越滚越大）。
# ============================================================
checkpointer = InMemorySaver()


# ============================================================
# 模块 5：组装 Agent
# 【作用】把 大脑(model) + 手脚(tools) + 人设(system_prompt) + 记忆(checkpointer)
#         装进一个可复用的 Agent 对象。
# 【对应】04 的整个 run_agent 函数定义——那个 for 循环被封装进这里，看不见但存在。
# 【注意】这一步【不执行】任何模型调用，只是"造好机器"。
# ============================================================
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


def show(result_messages, title):
    """打印本轮完整历史——即'解剖黑盒'，让你看到循环留下的脚印。"""
    print(f"\n{'=' * 60}\n{title}（本轮历史共 {len(result_messages)} 条消息）\n{'=' * 60}")
    for i, m in enumerate(result_messages):
        kind = type(m).__name__
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            names = [f"{tc['name']}({tc['args']})" for tc in tool_calls]
            print(f"  [{i}] {kind:<13} → 调用工具: {names}")
        else:
            preview = str(m.content)[:70].replace("\n", "\\n")
            print(f"  [{i}] {kind:<13} → {preview}")


def main():
    # 同一个 thread_id，让两轮对话共享记忆 —— 这是"记忆"生效的开关
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    # ---------- 模块 6：运行 —— 第一轮 ----------
    # 【作用】真正"按下启动键"：框架内部转循环(请求→调工具→回填→再请求)直到出答案。
    # 【对应】04 的 run_agent(client, task) 调用。
    q1 = "列出当前目录的文件，然后挑一个 .py 文件读一下，告诉我它大概是干嘛的。"
    print(f"【第 1 轮】用户：{q1}")
    r1 = agent.invoke({"messages": [{"role": "user", "content": q1}]}, config=config)
    show(r1["messages"], "第 1 轮")

    # ---------- 第二轮：故意问"第一轮的内容"，验证记忆 ----------
    q2 = "我上一轮让你列的是哪个目录？你还记得我让你读的是什么类型的文件吗？"
    print(f"\n【第 2 轮】用户：{q2}")
    # 本轮只传这一句新消息；框架会从 checkpointer 取回第 1 轮全部历史，
    # 拼在这句前面再【全量重发】给模型 —— 所以模型"记得"。
    r2 = agent.invoke({"messages": [{"role": "user", "content": q2}]}, config=config)
    show(r2["messages"], "第 2 轮（注意：历史里包含第 1 轮的全部内容 = 全量重发）")

    print(f"\n{'=' * 60}\n第 2 轮最终回答：\n{r2['messages'][-1].content}\n{'=' * 60}")


if __name__ == "__main__":
    main()
