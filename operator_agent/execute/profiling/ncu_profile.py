"""Parse an NCU raw CSV export and summarize the facts a profile receipt keeps.

The exporter (`ncu --csv --page raw`) prints application stdout and `==PROF==`
progress lines before the real header; the only reliable anchor is the row whose
first cell is `ID`. Units are read from the row under that header (the raw page
uses byte/ns, a re-export auto-scales to Mbyte/us), so the same parser handles
both. Values are normalized to bytes and nanoseconds.
"""
from __future__ import annotations

import csv
import io
import math


RAW_PROTOCOL = 'ncu-raw-csv-v1'
METRICS = (
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum',
    'lts__t_sector_hit_rate.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'gpu__time_duration.sum', 'launch__grid_size', 'launch__block_size', 'launch__sm_count',
    'launch__registers_per_thread', 'launch__shared_mem_per_block',
    'launch__occupancy_limit_registers', 'launch__occupancy_limit_shared_mem',
    'launch__occupancy_limit_warps', 'launch__occupancy_limit_blocks',
    'launch__occupancy_limit_barriers',
)
_BYTES = {'byte': 1.0, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'Tbyte': 1e12}
_TIME = {'ns': 1.0, 'us': 1e3, 'ms': 1e6, 's': 1e9}
_DECOY = ('vectorized_elementwise_kernel', 'elementwise_kernel')


class NcuReportError(ValueError):
    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


def _number(value):
    text = (value or '').replace(',', '').strip()
    return float(text) if text else None


def _scaled(value, unit, table):
    if value is None:
        return None
    if unit not in table:
        raise NcuReportError('unknown_unit', f'unknown unit {unit!r}')
    return value * table[unit]


def _base_unit(unit):
    """``byte/block`` and ``Kbyte/block`` both scale as bytes."""
    return (unit or '').split('/', 1)[0].strip()


def _integer(value):
    if value is None:
        return None
    rounded = round(value)
    return int(rounded) if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9) else value


def parse_raw_csv(text):
    """Return one dict per profiled launch, values in bytes/ns/percent/sectors."""
    rows = list(csv.reader(io.StringIO(text)))
    start = next((index for index, row in enumerate(rows)
                  if row and row[0].strip() == 'ID'), None)
    if start is None:
        raise NcuReportError('no_header', 'no ID header row in the NCU CSV')
    header = rows[start]
    units = rows[start + 1] if start + 1 < len(rows) else []
    columns = {name: index for index, name in enumerate(header)}

    def cell(row, name):
        index = columns.get(name)
        return row[index] if index is not None and index < len(row) else ''

    def value(row, name):
        return _number(cell(row, name))

    def unit(name):
        index = columns.get(name)
        return units[index] if index is not None and index < len(units) else ''

    range_index = next((index for index, name in enumerate(header)
                        if 'Push/Pop_Range' in name), None)
    launches = []
    for row in rows[start + 2:]:
        if not row or not row[0].strip().isdigit():
            continue
        duration = _scaled(value(row, 'gpu__time_duration.sum'),
                           unit('gpu__time_duration.sum'), _TIME)
        read = _scaled(value(row, 'dram__bytes_read.sum'),
                       unit('dram__bytes_read.sum'), _BYTES)
        write = _scaled(value(row, 'dram__bytes_write.sum'),
                        unit('dram__bytes_write.sum'), _BYTES)
        l2_read_sectors = value(row, 'lts__t_sectors_op_read.sum')
        l2_write_sectors = value(row, 'lts__t_sectors_op_write.sum')
        limits = {
            'registers': _integer(value(row, 'launch__occupancy_limit_registers')),
            'shared_memory': _integer(value(row, 'launch__occupancy_limit_shared_mem')),
            'warps': _integer(value(row, 'launch__occupancy_limit_warps')),
            'blocks': _integer(value(row, 'launch__occupancy_limit_blocks')),
            'barriers': _integer(value(row, 'launch__occupancy_limit_barriers')),
        }
        bounded = [v for v in limits.values() if v is not None]
        shared_bytes = _scaled(
            value(row, 'launch__shared_mem_per_block'),
            _base_unit(unit('launch__shared_mem_per_block')), _BYTES)
        launches.append(dict(
            kernel=cell(row, 'Kernel Name').strip(),
            grid=cell(row, 'Grid Size').strip(),
            block=cell(row, 'Block Size').strip(),
            stream=cell(row, 'Stream').strip(),
            duration_ns=duration,
            duration_ms=(duration / 1e6 if duration is not None else None),
            dram_read_bytes=read,
            dram_write_bytes=write,
            l2_read_sectors=l2_read_sectors,
            l2_write_sectors=l2_write_sectors,
            l2_read_bytes=(l2_read_sectors * 32 if l2_read_sectors is not None else None),
            l2_write_bytes=(l2_write_sectors * 32 if l2_write_sectors is not None else None),
            l2_hit_pct=value(row, 'lts__t_sector_hit_rate.pct'),
            warps_pct=value(row, 'sm__warps_active.avg.pct_of_peak_sustained_active'),
            long_scoreboard_pct=value(
                row, 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),
            registers_per_thread=_integer(value(row, 'launch__registers_per_thread')),
            shared_memory_bytes_per_block=shared_bytes,
            registers_limit_blocks_per_sm=limits['registers'],
            shared_memory_limit_blocks_per_sm=limits['shared_memory'],
            warps_limit_blocks_per_sm=limits['warps'],
            block_limit_blocks_per_sm=limits['blocks'],
            barriers_limit_blocks_per_sm=limits['barriers'],
            occupancy_limit_blocks_per_sm=min(bounded) if bounded else None,
            resource_limits_blocks_per_sm=limits,
            nvtx=(row[range_index] if range_index is not None
                  and range_index < len(row) else ''),
        ))
    if not launches:
        raise NcuReportError('no_launches', 'the NCU export has no kernel rows')
    return launches


def summarize(launches, *, range_name='candidate', logical_bytes=None):
    in_range = [row for row in launches if range_name in (row.get('nvtx') or '')]
    decoy = [row for row in launches
             if any(marker in row['kernel'] for marker in _DECOY)]
    total_read = sum(row['dram_read_bytes'] or 0 for row in launches)
    total_write = sum(row['dram_write_bytes'] or 0 for row in launches)
    total_ns = sum(row['duration_ns'] or 0 for row in launches)
    totals = dict(dram_read_bytes=total_read, dram_write_bytes=total_write,
                  duration_ns=total_ns, duration_ms=total_ns / 1e6,
                  launches=len(launches))
    if logical_bytes:
        totals['logical_bytes'] = logical_bytes
        totals['dram_read_x_logical'] = total_read / logical_bytes
        totals['dram_write_x_logical'] = total_write / logical_bytes
    return dict(range_name=range_name, launches=launches, totals=totals,
                in_range=len(in_range), decoy_seen=bool(decoy),
                decoy_kernels=[row['kernel'] for row in decoy],
                range_ok=len(in_range) == len(launches))


def compare(original, reexport, *, rel_tol=1e-9, abs_tol=1.0):
    """The same report re-exported must reproduce kernel names, bytes and time."""
    if len(original) != len(reexport):
        return dict(match=False, reasons=[
            f'launch count {len(original)} != {len(reexport)}'])
    reasons = []
    for index, (left, right) in enumerate(zip(original, reexport)):
        if left['kernel'] != right['kernel']:
            reasons.append(f'{index}: kernel {left["kernel"]!r} != {right["kernel"]!r}')
        for key in ('dram_read_bytes', 'dram_write_bytes', 'duration_ns'):
            a, b = left.get(key), right.get(key)
            if a is None or b is None or not math.isclose(
                    a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                reasons.append(f'{index}: {key} {a} != {b}')
        for key in ('registers_per_thread', 'shared_memory_bytes_per_block',
                    'registers_limit_blocks_per_sm', 'shared_memory_limit_blocks_per_sm',
                    'warps_limit_blocks_per_sm', 'block_limit_blocks_per_sm',
                    'barriers_limit_blocks_per_sm', 'occupancy_limit_blocks_per_sm'):
            a, b = left.get(key), right.get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or not math.isclose(
                    a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                reasons.append(f'{index}: {key} {a} != {b}')
    return dict(match=not reasons, reasons=reasons)


def cache_conditions(metrics=None, baseline_performance=None):
    """Label the profile replay and the independent final-timing cache regimes."""
    metrics = metrics or {}
    perf = baseline_performance or {}
    flush = perf.get('l2_flush_bytes')
    if flush is None:
        flush = metrics.get('l2_flush_bytes')
    channel = perf.get('timing_channel') or metrics.get('timing_channel')
    protocol = perf.get('timing_protocol') or metrics.get('timing_protocol')
    return dict(
        profile=dict(
            replay='kernel', cache_control='none', clock_control='none',
            note='NCU kernel replay with uncontrolled GPU caches/clocks; '
                 'not directly comparable to final L3'),
        final=dict(
            timing_channel=channel, timing_protocol=protocol,
            l2_flush_bytes=flush,
            source='baseline.json' if perf else 'l3-metrics',
            note=('independent L3 production timing; L2 flushed per run'
                  if flush is not None else
                  'independent L3 production timing; L2 flush not in receipt')),
    )
