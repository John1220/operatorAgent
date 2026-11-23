"""v0.8.2: NCU raw CSV parsing, range/decoy checks and re-export comparison."""
from pathlib import Path

import pytest

from operator_agent.execute import ncu_profile as np


# 按仓库根定位而不是数层级：这个文件在 tests/unit/profile/ 下，数层级会在
# 目录调整时悄悄指错地方（本次搬运就踩过一次）。
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'benchmarks').is_dir())
SPIKE = ROOT/'benchmarks'/'experiments'/'v082'/'nvtx-spike'
B = 32768 * 6144 * 2


def test_dual_seed_profile_summary():
    launches = np.parse_raw_csv((SPIKE/'dual-seed.csv').read_text())
    summary = np.summarize(launches, range_name='candidate', logical_bytes=B)
    assert summary['range_ok'] is True and summary['decoy_seen'] is False
    assert summary['totals']['launches'] == 2
    assert summary['totals']['dram_read_x_logical'] == pytest.approx(2.001, abs=0.01)
    assert [row['kernel'] for row in launches] == ['_stats_kernel', '_normalize_kernel']
    assert launches[1]['l2_write_bytes'] == pytest.approx(launches[1]['l2_write_sectors'] * 32)


def test_reexport_reproduces_the_same_values():
    original = np.parse_raw_csv((SPIKE/'dual-seed.csv').read_text())
    reexport = np.parse_raw_csv((SPIKE/'dual-seed.reimport.csv').read_text())
    assert np.compare(original, reexport) == dict(match=True, reasons=[])


def test_decoy_and_out_of_range_are_visible():
    launches = np.parse_raw_csv((SPIKE/'dual-seed.csv').read_text())
    decoy = dict(launches[0], kernel='vectorized_elementwise_kernel')
    summary = np.summarize([*launches, decoy], range_name='candidate', logical_bytes=B)
    assert summary['decoy_seen'] is True
    assert summary['decoy_kernels'] == ['vectorized_elementwise_kernel']
    outside = dict(launches[0], nvtx='')
    summary = np.summarize([outside], range_name='candidate', logical_bytes=B)
    assert summary['range_ok'] is False


def test_missing_header_is_a_typed_error():
    with pytest.raises(np.NcuReportError, match='no ID header'):
        np.parse_raw_csv('==PROF== nothing here\n')


def _occupancy_csv(shared_unit, shared_value):
    header = [
        'ID', 'Kernel Name', 'Grid Size', 'Block Size', 'Stream',
        'gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
        'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum',
        'lts__t_sector_hit_rate.pct',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
        'launch__registers_per_thread', 'launch__shared_mem_per_block',
        'launch__occupancy_limit_registers', 'launch__occupancy_limit_shared_mem',
        'launch__occupancy_limit_warps', 'launch__occupancy_limit_blocks',
        'launch__occupancy_limit_barriers',
        'domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg',
    ]
    units = ['', '', '', '', '',
             'ms', 'byte', 'byte', 'sector', 'sector', '%', '%', '%',
             'register/thread', shared_unit, 'block', 'block', 'block', 'block',
             'block', '']
    data = ['0', '_attn_fwd', '64x16x1', '128x1x1', '7',
            '1.205024', '45481728', '9927936', '49380446', '788405',
            '88.34', '12.44', '11.04', '168', str(shared_value),
            '3', '2', '16', '32', '32', '0  "candidate"']
    return '\n'.join(','.join(row) for row in (header, units, data)) + '\n'


def test_occupancy_units_and_limits_normalize_both_exports():
    raw = np.parse_raw_csv(_occupancy_csv('byte/block', 82944))
    reexport = np.parse_raw_csv(_occupancy_csv('Kbyte/block', 82.944))
    for rows in (raw, reexport):
        row = rows[0]
        assert row['registers_per_thread'] == 168
        assert row['shared_memory_bytes_per_block'] == 82944
        assert row['registers_limit_blocks_per_sm'] == 3
        assert row['shared_memory_limit_blocks_per_sm'] == 2
        assert row['occupancy_limit_blocks_per_sm'] == 2
        assert row['resource_limits_blocks_per_sm'] == dict(
            registers=3, shared_memory=2, warps=16, blocks=32, barriers=32)
    assert np.compare(raw, reexport) == dict(match=True, reasons=[])
