"""一键运行 steps 里的脚本，自动从 .env 读取配置。

用法（在 nano-agent 目录下）：
    .venv\\Scripts\\python.exe run.py steps/01_hello_llm.py
    .venv\\Scripts\\python.exe run.py steps/04_agent_loop.py

这样不用每次手动设环境变量，也不会把密钥写进命令行历史。
"""

import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

# 只读取本文件旁边的 .env，换目录也能找到
load_dotenv(Path(__file__).with_name(".env"))


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python run.py steps/01_hello_llm.py")
        return 1

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"找不到文件: {target}")
        return 1

    # 把目标脚本当作主程序执行，它内部的 __main__ 判断才会生效
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
