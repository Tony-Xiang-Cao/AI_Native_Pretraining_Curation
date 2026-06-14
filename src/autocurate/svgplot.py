"""Tiny pure-stdlib SVG plotting (no matplotlib) for the paper figures.

Just enough to draw grouped bar charts and multi-line charts deterministically,
so figures regenerate bit-for-bit in any environment.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _hdr(w: int, h: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>')


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x, y, s, size=13, anchor="middle", color="#111", weight="normal") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}">{_esc(s)}</text>')


def bar_chart(groups: List[str], series: Dict[str, Sequence[float]], title: str,
              ymax: float = 1.0, w: int = 720, h: int = 380) -> str:
    ml, mr, mt, mb = 60, 20, 50, 60
    pw, ph = w - ml - mr, h - mt - mb
    out = [_hdr(w, h), _text(w / 2, 28, title, 16, weight="bold")]
    # axes
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#888"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#888"/>')
    for t in range(0, 6):
        yv = ymax * t / 5
        y = mt + ph - ph * (yv / ymax)
        out.append(f'<line x1="{ml}" y1="{y}" x2="{ml+pw}" y2="{y}" stroke="#eee"/>')
        out.append(_text(ml - 8, y + 4, f"{yv:.1f}", 11, "end"))
    n_groups = len(groups)
    n_series = len(series)
    gw = pw / n_groups
    bw = gw * 0.8 / max(1, n_series)
    for gi, g in enumerate(groups):
        gx = ml + gi * gw + gw * 0.1
        for si, (sname, vals) in enumerate(series.items()):
            v = vals[gi]
            bh = ph * (v / ymax)
            x = gx + si * bw
            y = mt + ph - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" height="{bh:.1f}" '
                       f'fill="{_COLORS[si % len(_COLORS)]}"/>')
            out.append(_text(x + bw * 0.45, y - 4, f"{v:.2f}", 10))
        out.append(_text(ml + gi * gw + gw / 2, mt + ph + 18, g, 12))
    # legend
    lx = ml + 10
    for si, sname in enumerate(series):
        out.append(f'<rect x="{lx}" y="{h-28}" width="12" height="12" fill="{_COLORS[si%len(_COLORS)]}"/>')
        out.append(_text(lx + 16, h - 18, sname, 12, "start"))
        lx += 30 + len(sname) * 7
    out.append("</svg>")
    return "".join(out)


def line_chart(lines: Dict[str, List[Tuple[float, float]]], title: str,
               xlabel: str, ylabel: str, ymax: float = 1.0, ymin: float = 0.0,
               xmax: float = None, w: int = 720, h: int = 380,
               markers: List[Tuple[float, str]] = None) -> str:
    ml, mr, mt, mb = 64, 20, 50, 56
    pw, ph = w - ml - mr, h - mt - mb
    allx = [x for pts in lines.values() for x, _ in pts]
    xmx = xmax if xmax is not None else (max(allx) if allx else 1.0)
    xmx = xmx or 1.0
    out = [_hdr(w, h), _text(w / 2, 28, title, 16, weight="bold")]
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#888"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#888"/>')
    for t in range(0, 6):
        yv = ymin + (ymax - ymin) * t / 5
        y = mt + ph - ph * ((yv - ymin) / (ymax - ymin))
        out.append(f'<line x1="{ml}" y1="{y}" x2="{ml+pw}" y2="{y}" stroke="#eee"/>')
        out.append(_text(ml - 8, y + 4, f"{yv:.2f}", 11, "end"))
    out.append(_text(ml + pw / 2, h - 14, xlabel, 13))
    out.append(f'<text x="16" y="{mt+ph/2}" font-size="13" text-anchor="middle" '
               f'transform="rotate(-90 16 {mt+ph/2})">{_esc(ylabel)}</text>')

    def px(x): return ml + pw * (x / xmx)
    def py(y): return mt + ph - ph * ((y - ymin) / (ymax - ymin))

    if markers:
        for mx, label in markers:
            x = px(mx)
            out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" '
                       f'stroke="#999" stroke-dasharray="4 3"/>')
            out.append(_text(x, mt - 4, label, 11, "middle", "#666"))
    for si, (name, pts) in enumerate(lines.items()):
        c = _COLORS[si % len(_COLORS)]
        d = " ".join(f"{'M' if i==0 else 'L'} {px(x):.1f} {py(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.4" fill="{c}"/>')
    lx = ml + 10
    for si, name in enumerate(lines):
        out.append(f'<rect x="{lx}" y="{mt+4}" width="12" height="12" fill="{_COLORS[si%len(_COLORS)]}"/>')
        out.append(_text(lx + 16, mt + 14, name, 12, "start"))
        lx += 40 + len(name) * 7
    out.append("</svg>")
    return "".join(out)
