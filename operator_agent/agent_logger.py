"""Agent 执行日志模块
将 Agent 执行过程的所有输出同时写入控制台和日志文件

日志目录: <project_root>/logs/
日志文件: <timestamp>_<task_id>.log

使用方式:
  # 方式1: Conductor 自动使用 (内部集成)
  conductor = Conductor(...)
  artifact = conductor.execute(task, inputs)  # 自动记录日志

  # 方式2: 手动使用上下文管理器
  from operator_agent.agent_logger import capture_execution
  with capture_execution(task_id="my_task", log_dir="logs") as log_path:
      ... agent 代码 ...

  # 方式3: 直接使用 logger
  from operator_agent.agent_logger import get_logger
  logger = get_logger()
  logger.info("执行中...")
"""

import sys
import os
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class TeeWriter:
    """同时写入原始 stdout 和日志文件的分流器"""

    def __init__(self, original_stream, log_file_handle):
        self._original = original_stream
        self._log_file = log_file_handle
        self._buffer = io.StringIO()

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)
        self._buffer.write(data)
        self._original.flush()
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()

    @property
    def captured_text(self) -> str:
        return self._buffer.getvalue()


@contextmanager
def capture_execution(
    task_id: str = "",
    log_dir: Optional[str] = None,
    log_name: Optional[str] = None,
    operator: str = "",
    hardware: str = "",
):
    """
    上下文管理器: 捕获执行期间所有 stdout/stderr 输出到日志文件

    Args:
        task_id: 任务 ID (用于日志文件名)
        log_dir: 日志目录 (默认 <project_root>/logs/)
        log_name: 自定义日志文件名 (不含扩展名)
        operator: 算子名称 (如 matmul, layernorm)，用于日志文件名
        hardware: 目标硬件 (如 nvidia_gpu)，用于日志文件名

    Yields:
        Path: 日志文件路径

    Example:
        with capture_execution(task_id="abc123", operator="matmul", hardware="nvidia_gpu") as log_path:
            print("这行会同时输出到控制台和日志文件")
        # 日志文件名: 20260802_153000_matmul_nvidia_gpu_ab12cd34.log
    """
    # 确定日志目录
    if log_dir:
        log_dir = Path(log_dir)
    else:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 自动清理旧日志，只保留最新10个
    deleted = cleanup_old_logs(log_dir, keep_count=10)
    if deleted > 0:
        print(f"[Log] 自动清理 {deleted} 个旧日志文件")

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tid = task_id[:8] if task_id else "session"
    if log_name:
        log_filename = f"{log_name}.log"
    else:
        # 构建文件名: timestamp_operator_hardware_tid.log
        parts = [timestamp]
        if operator:
            parts.append(operator)
        if hardware:
            parts.append(hardware)
        parts.append(tid)
        log_filename = "_".join(parts) + ".log"
    log_path = log_dir / log_filename

    # 打开日志文件
    log_file = open(log_path, "w", encoding="utf-8")

    # 写入日志头
    header = f"{'='*60}\n"
    header += f"Agent 执行日志\n"
    header += f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"  任务ID: {task_id}\n"
    header += f"  日志文件: {log_path}\n"
    header += f"{'='*60}\n\n"
    log_file.write(header)
    log_file.flush()

    # 保存原始流
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 创建分流器
    tee_out = TeeWriter(original_stdout, log_file)
    tee_err = TeeWriter(original_stderr, log_file)

    sys.stdout = tee_out
    sys.stderr = tee_err

    try:
        yield log_path
    except Exception as e:
        # 捕获异常信息写入日志
        err_msg = f"\n{'!'*60}\n"
        err_msg += f"执行异常: {type(e).__name__}: {e}\n"
        err_msg += f"{'!'*60}\n"
        tee_out.write(err_msg)
        raise
    finally:
        # 写入日志尾
        footer = f"\n{'='*60}\n"
        footer += f"执行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"日志文件: {log_path}\n"
        footer += f"{'='*60}\n"
        log_file.write(footer)

        # 恢复原始流
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        log_file.close()

        # 输出日志路径
        print(f"\n[Log] 执行日志已保存到: {log_path}")


def get_logger(name: str = "operator_agent") -> logging.Logger:
    """获取标准 Python logger (用于结构化日志)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
    return logger


def get_latest_logs(log_dir: Optional[str] = None, n: int = 5) -> list:
    """获取最近的 N 个日志文件路径"""
    if log_dir:
        log_dir = Path(log_dir)
    else:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    if not log_dir.exists():
        return []
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[:n]


def read_log(log_path: str) -> str:
    """读取日志文件内容"""
    return Path(log_path).read_text(encoding="utf-8")


def cleanup_old_logs(log_dir: Optional[str] = None, keep_count: int = 10) -> int:
    """
    清理旧日志文件，只保留最新的 N 个

    Args:
        log_dir: 日志目录 (默认 <project_root>/logs/)
        keep_count: 保留的日志文件数量

    Returns:
        int: 删除的日志文件数量
    """
    if log_dir:
        log_dir = Path(log_dir)
    else:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    if not log_dir.exists():
        return 0

    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(logs) <= keep_count:
        return 0

    to_delete = logs[keep_count:]
    deleted_count = 0
    for log_file in to_delete:
        try:
            log_file.unlink()
            deleted_count += 1
        except Exception:
            pass
    return deleted_count
