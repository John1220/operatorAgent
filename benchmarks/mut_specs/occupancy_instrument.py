import shutil, subprocess, sys, tempfile
from pathlib import Path
R = Path(__file__).resolve().parents[2]   # 仓库根: 相对定位 (公开仓不许出现绝对路径); F = R / "operator_agent/decide/escalation.py"
M = [
 ("M1 smem 未知也敢报", """    for value in (tile_smem_bytes, num_warps, max_threads_per_sm, shared_mem_per_sm_kb):""",
  """    for value in (tile_smem_bytes, num_warps, max_threads_per_sm):"""),
 ("M2 smem 更紧也照报", """    if ctas_by_smem < ctas_by_threads:
        return ()      # 更紧的是 smem 侧 ⇒ 那是 smem-bound, 不由本条认领""", """    if False:
        return ()"""),
 ("M3 门槛 off-by-one", """    if ctas_by_threads > OCCUPANCY_FLOOR_CTAS:""", """    if ctas_by_threads >= OCCUPANCY_FLOOR_CTAS:"""),
 ("M4 布尔当整数", """        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:""",
  """        if not isinstance(value, int) or value <= 0:"""),
 ("M5 不写出处", """+ HardwareCaps.max_threads_per_sm""", ""),
]
def run():
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/unit/test_occupancy_instrument.py", "-q"],
                       cwd=str(R), capture_output=True, text=True)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]
rows = []
for label, old, new in M:
    b = Path(tempfile.gettempdir()) / "occ.bak"; shutil.copy2(F, b)
    try:
        t = F.read_text(encoding="utf-8")
        if t.count(old) != 1:
            rows.append((label, "INVALID", f"锚点 {t.count(old)} 次")); continue
        F.write_text(t.replace(old, new, 1), encoding="utf-8")
        code, tail = run(); rows.append((label, "RED" if code else "GREEN(逃逸!)", tail))
    finally:
        shutil.copy2(b, F); b.unlink()
for l, r, t in rows: print(f"{l:<22} {r:<14} {t}")
g = [r for r in rows if r[1].startswith("GREEN")]; iv = [r for r in rows if r[1] == "INVALID"]
print(f"\n合计 {len(rows)}: 红 {len(rows)-len(g)-len(iv)}, 逃逸 {len(g)}, 无效 {len(iv)}")
