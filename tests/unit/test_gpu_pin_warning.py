"""P2-3 (docs/design/generation-side-review.md): 选卡意图被静默覆盖必须有 WARN

`SSHExecutor._pinned_gpu()` 按利用率自动选卡并下发**自己的**选择, 本地
`OPERATOR_AGENT_GPU` 只对远端 runner 生效 —— 控制端从不读它。doc 49 §851
的事故形态: 本地设 2、实际跑 7、全程无报错。本文件钉三条:
  1. 不一致必须 WARN (消除「我以为我在测卡 2」的实验污染);
  2. 一致 / 未设置 / 非法卡号各归各位: 一致与未设置**不得**多打 ⚠️
     (狼来了效应会让真警告被忽略), 非法卡号要说一句而不是静默吞掉;
  3. WARN 只提醒, 不改选卡行为 —— 「按利用率选卡」是共享机器上的既定设计,
     不由这个环境变量推翻 (钉一条同机同选择的对照)。
"""

from operator_agent.execute.remote_executor import SSHExecutor

#: nvidia-smi csv 的真实形状 (index, memory.free, utilization.gpu)。
#: 两张卡里选 1 —— util 最低优先 (利用率优先于显存余量)。
_SMI_OUTPUT = "0, 80000, 95\n1, 90000, 5\n"


def _ssh(monkeypatch, smi_output=_SMI_OUTPUT, env_gpu=None):
    """构造不碰网络的 SSHExecutor; `_ssh_exec` 只应答 nvidia-smi 那一条"""
    ex = SSHExecutor(host="pin.invalid", platform="nvidia", work_dir="/tmp/pin_wd")
    ex._deployed = True  # 与 test_hardware_probe_honesty 同款: 构造函数只做字符串拼装
    monkeypatch.setattr(ex, "_ssh_exec", lambda cmd, *a, **k: smi_output)
    if env_gpu is None:
        monkeypatch.delenv("OPERATOR_AGENT_GPU", raising=False)
    else:
        monkeypatch.setenv("OPERATOR_AGENT_GPU", env_gpu)
    return ex


def test_mismatch_prints_warning(monkeypatch, capsys):
    """本地设 2、自动选到 1 → 必须有一行 ⚠️ 同时说清「设了什么」与「实际用哪张」"""
    ex = _ssh(monkeypatch, env_gpu="2")
    assert ex._pinned_gpu() == 1
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "OPERATOR_AGENT_GPU=2" in out
    assert "GPU 1" in out


def test_match_prints_no_warning(monkeypatch, capsys):
    """本地设 1、自动也选到 1 → 一致不是事故, 不打 ⚠️"""
    ex = _ssh(monkeypatch, env_gpu="1")
    assert ex._pinned_gpu() == 1
    assert "⚠️" not in capsys.readouterr().out


def test_unset_prints_no_warning(monkeypatch, capsys):
    """未设置 → 只有常规选卡行; 用户没表达意图就没有可被覆盖的东西"""
    ex = _ssh(monkeypatch, env_gpu=None)
    assert ex._pinned_gpu() == 1
    out = capsys.readouterr().out
    assert "⚠️" not in out
    assert "固定使用 GPU 1" in out


def test_invalid_card_number_is_reported(monkeypatch, capsys):
    """非法卡号 (非数字) → 要说一句「已忽略」, 不能静默吞掉用户的笔误"""
    ex = _ssh(monkeypatch, env_gpu="zero")
    assert ex._pinned_gpu() == 1
    out = capsys.readouterr().out
    assert "不是合法卡号" in out


def test_warning_does_not_change_selection(monkeypatch):
    """WARN 只提醒: 同一台机器上, 设与不设 env 选出的卡必须一致"""
    a = _ssh(monkeypatch, env_gpu=None)
    b = _ssh(monkeypatch, env_gpu="0")  # 0 是会被选卡否决的意图 (自动选 1)
    assert a._pinned_gpu() == b._pinned_gpu() == 1
