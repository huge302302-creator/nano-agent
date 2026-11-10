# 分步理解 Agent（建议按顺序运行）

这 4 个脚本把一个 Agent 拆开给你看。每一步只增加一个新概念，
运行完 4 步，你就已经亲手实现了一个最小 Agent。

所有命令都在**项目根目录**执行（因为脚本里会读根目录的 `README.md`）。

## 前置条件

```bash
pip install -r requirements.txt
```

并设置好环境变量：

```bash
# PowerShell
$env:OPENAI_API_KEY="your-key-here"
# CMD
set OPENAI_API_KEY=your-key-here
# Git Bash / macOS / Linux
export OPENAI_API_KEY="your-key-here"
```

如果使用兼容 OpenAI 协议的其他服务，再设置 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

## 四个步骤

| 步骤 | 文件 | 新概念 | 你会观察到 |
|---|---|---|---|
| 1 | `01_hello_llm.py` | `messages` 和三种 role | 模型只会说话，不会做事 |
| 2 | `02_watch_tool_call.py` | `tools` 菜单和 `tool_calls` | 模型只是**请求**调用工具，文件并没有被读 |
| 3 | `03_execute_one_tool.py` | 执行工具 + `role="tool"` 回填 | 模型第二次请求时才真正“看到”文件内容 |
| 4 | `04_agent_loop.py` | `while` 循环 + 退出条件 | 模型自己决定调用几次工具 |

## 运行

```bash
python steps/01_hello_llm.py
python steps/02_watch_tool_call.py
python steps/03_execute_one_tool.py
python steps/04_agent_loop.py
```

第 4 步也可以换成自己的任务：

```bash
python steps/04_agent_loop.py "读一下 agent.py，说明它的核心循环在第几行"
```

## 每一步到底新增了什么

### 01：只有对话

```python
messages = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "什么是 AI Agent？"},
]
response = client.chat.completions.create(model=MODEL, messages=messages)
```

没有 `tools`，模型只能生成文本。这是所有 Agent 的起点。

### 02：给模型一份工具菜单

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,          # 新增
)
```

模型返回的 `message.tool_calls` 长这样：

```python
[
    ToolCall(
        id="call_abc123",
        function=Function(
            name="read_file",
            arguments='{"path": "README.md"}',
        ),
    )
]
```

注意：`arguments` 是 JSON **字符串**，而且**没有任何文件被读取**。

### 03：真正执行，并把结果交还

```python
# 1. 保存模型带 tool_calls 的回复
messages.append(message)

# 2. 执行工具
result = read_file(**json.loads(tool_call.function.arguments))

# 3. 用 tool 角色回填，id 必须对应
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": result,
})

# 4. 再请求一次，模型才能基于结果回答
response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
```

### 04：把上面三步放进循环

```python
for _ in range(max_iterations):
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:      # 退出条件
        return message.content

    for tool_call in message.tool_calls:
        result = FUNCTIONS[tool_call.function.name](**args)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
```

## 接下来

读完这 4 步，再看根目录的 [`agent.py`](../agent.py)。它只是在这个循环上多了：

- `write_file` 和 `execute_bash` 两个能修改系统的工具
- 危险操作前的人工确认（`confirm_tool_call`）
- 坏 JSON、未知工具、工具异常的保护
- 更完整的注释

然后可以继续看：

- [`agent-plus.py`](../agent-plus.py)：任务规划（plan）与长期记忆（memory）
- [`agent-claudecode.py`](../agent-claudecode.py)：规则、技能、MCP 工具加载

## 常见问题

**报错 401 / Incorrect API key**
检查 `OPENAI_API_KEY` 是否设置成功。PowerShell 里设置后需要在新终端或同一会话中运行。

**报错 model not found / does not support tools**
你使用的模型不支持 function calling。换一个支持工具调用的模型，或设置 `OPENAI_MODEL`。

**02/03 报错找不到 README.md**
请在项目根目录运行脚本，而不是 `cd steps` 之后再运行。
