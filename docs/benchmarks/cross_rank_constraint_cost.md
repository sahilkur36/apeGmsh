# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-06-11 02:07:18 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.005 | 0.008 | 318.7 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.006 | 0.013 | 324.2 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.004 | 0.008 | 324.2 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.005 | 0.011 | 324.4 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.006 | 0.009 | 324.4 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.006 | 0.012 | 324.4 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.033 | 0.089 | 383.2 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.044 | 0.123 | 414.4 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.035 | 0.085 | 414.4 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.202 | 0.124 | 414.6 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.042 | 0.085 | 414.6 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.057 | 0.122 | 416.3 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.481 | 1.047 | 988.3 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.619 | 1.330 | 1262.0 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.522 | 1.010 | 1262.0 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.732 | 1.439 | 1263.7 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.585 | 0.920 | 1263.7 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.779 | 1.299 | 1263.7 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 4.025 | 10.610 | 6977.9 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 6.052 | 15.601 | 9715.5 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 4.881 | 10.020 | 9715.5 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 6.921 | 14.054 | 9715.5 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 5.613 | 11.327 | 9715.5 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 9.993 | 18.401 | 9715.5 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
