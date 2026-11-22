"""Figure 2: city rankings, empirical comparison, population groups and maps."""

from pathlib import Path
import hashlib
import io
import json
import struct
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.utils import ImageReader
from utils.plot_style import publication_font, publication_poppler

W, H = 518.4, 435.0
B_RECT = (270.0, 183.0, 84.0, 220.0)
C_RECT = (392.0, 317.0, 115.0, 86.0)
D_RECT = (392.0, 183.0, 115.0, 94.0)
EMPIRICAL = [0.2177, 0.2063, 0.1890, 0.1386]
SIMULATED = [0.2072, 0.2026, 0.1623, 0.1547]
CITY_LABELS = ["Beijing", "Shenzhen", "Wuhan", "Xiamen"]
COLORS = ["#C9DBEA", "#A8C5DB", "#80A8C5", "#507D9C"]
PURPLE, ORANGE = "#482173", "#E76F51"
MAP_TOP = 130.0
MAP_BOTTOM = 67.95121951219471
MAP_HEIGHT = MAP_TOP - MAP_BOTTOM
SPECS = [
    ("fangchenggang", "Fangchenggang", 1, "Guangxi/fangchenggang_pop.shp", 67.0),
    ("weinan", "Weinan", 250, "Shaanxi/weinan_pop.shp", 198.0),
    ("yueyang", "Yueyang", 251, "Hunan/yueyang_pop.shp", 315.0),
    ("danzhou", "Danzhou", 400, "Hainan/danzhou_pop.shp", 420.0),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(c, x, y, value, size=7.5, align="center", bold=False, color="#222222"):
    c.setFillColor(HexColor(color))
    c.setFont("Arial-Bold" if bold else "Arial", size)
    {"center": c.drawCentredString, "left": c.drawString, "right": c.drawRightString}[
        align
    ](x, y, value)


def txt(c, x, y, s, size=7, align="center", bold=False):
    return text(c, x, y, s, size, align, bold)


def panel(c, x, y, label):
    txt(c, x, y, label, 11, "left", True)


def rd_bu_r(value, vmin, vmax):
    # Matplotlib's RdBu_r 11 anchors and default 256-entry linear LUT.
    anchors = [
        "#053061",
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
        "#67001f",
    ]
    rgb = np.array(
        [[HexColor(h).red, HexColor(h).green, HexColor(h).blue] for h in anchors]
    )
    u = min(int(np.clip((value - vmin) / (vmax - vmin), 0, 1) * 256), 255) / 255 * 10
    lo = min(int(u), 9)
    v = rgb[lo] * (1 - (u - lo)) + rgb[lo + 1] * (u - lo)
    return Color(*v)


def axes(c, x, y, w, h, lo, hi, ticks):
    c.setStrokeColor(HexColor("#222222"))
    c.setLineWidth(0.65)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    for value in ticks:
        yy = y + (value - lo) / (hi - lo) * h
        c.line(x - 2.5, yy, x, yy)
        txt(c, x - 4, yy - 2, f"{value:.2f}", 6, "right")
    c.saveState()
    c.translate(x - 21, y + h / 2)
    c.rotate(90)
    txt(c, 0, 0, "η", 8)
    c.restoreState()


def read_dbf(path):
    """Read the numeric DBF attributes, retaining the physical record order."""
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 4)[0]
    header, length = struct.unpack_from("<HH", raw, 8)
    fields = []
    pos = 32
    while raw[pos] != 13:
        field = raw[pos : pos + 32]
        name = field[:11].split(b"\0", 1)[0].decode("ascii")
        fields.append((name, chr(field[11]), field[16]))
        pos += 32
    records = []
    for i in range(count):
        item = raw[header + i * length : header + (i + 1) * length]
        offset, record = 1, {"deleted": item[:1] == b"*"}
        for name, kind, size in fields:
            value = item[offset : offset + size].strip()
            if kind in ("N", "F"):
                record[name] = float(value) if value else float("nan")
            else:
                record[name] = value.decode("utf-8", errors="replace")
            offset += size
        records.append(record)
    return records, fields


def read_polygons(path):
    """Read ESRI Polygon/PolygonZ/PolygonM XY rings in record order."""
    raw = path.read_bytes()
    assert struct.unpack_from(">i", raw, 0)[0] == 9994
    assert struct.unpack_from(">i", raw, 24)[0] * 2 == len(raw)
    polygons, pos = [], 100
    while pos < len(raw):
        record_num, words = struct.unpack_from(">ii", raw, pos)
        body = raw[pos + 8 : pos + 8 + words * 2]
        pos += 8 + words * 2
        kind = struct.unpack_from("<i", body, 0)[0]
        if kind == 0:
            polygons.append(None)
            continue
        assert kind in (5, 15, 25), (path, record_num, kind)
        parts, points = struct.unpack_from("<ii", body, 36)
        starts = list(struct.unpack_from("<" + "i" * parts, body, 44)) + [points]
        xy = (
            np.frombuffer(body, dtype="<f8", count=points * 2, offset=44 + parts * 4)
            .reshape(-1, 2)
            .copy()
        )
        polygons.append([xy[starts[i] : starts[i + 1]] for i in range(parts)])
    return polygons


def population_map(key, relative, data_root):
    shape_path = Path(data_root) / "country" / relative
    dest = shape_path.parent
    attrs, fields = read_dbf(shape_path.with_suffix(".dbf"))
    polygons = read_polygons(shape_path)
    assert len(attrs) == len(polygons)
    value_key = next(name for name, kind, size in fields if name.lower() == "value")
    cells = [
        (p, a[value_key])
        for p, a in zip(polygons, attrs)
        if p is not None
        and not a["deleted"]
        and np.isfinite(a[value_key])
        and a[value_key] > 0
    ]
    assert cells
    values = np.array([v for p, v in cells])
    minimum, maximum = float(values.min()), float(values.max())
    relative_values = (
        (values - minimum) / (maximum - minimum)
        if not np.isclose(minimum, maximum)
        else np.full(len(values), 0.5)
    )
    xy = np.concatenate([ring for rings, value in cells for ring in rings])
    bounds = [*xy.min(axis=0), *xy.max(axis=0)]
    metadata = dict(
        city=key,
        source_records=len(attrs),
        retained_cells=len(cells),
        excluded_nonpositive_or_invalid=len(attrs) - len(cells),
        field=value_key,
        value_min=minimum,
        value_max=maximum,
        bounds=[float(v) for v in bounds],
        hashes={p.name: sha(p) for p in [shape_path, shape_path.with_suffix(".dbf")]},
    )
    return cells, relative_values, bounds, metadata


def merged_ranking(c, top, bottom, vmin, vmax, maximum):
    x, y, width, height = B_RECT
    gap = 26.0
    step = (height - gap) / 10
    positions = []
    for group, rows in enumerate([top, bottom]):
        for i, row in enumerate(rows.itertuples()):
            yy = y + height - (group * 5 + i + 0.5) * step - group * gap
            c.setFillColor(rd_bu_r(row.eta, vmin, vmax))
            length = float(row.eta / maximum * width)
            c.rect(x, yy - 0.39 * step, length, 0.78 * step, fill=1, stroke=0)
            name = "Qiqihar" if row.city == "qiqihaer" else row.city.title()
            txt(c, x - 3, yy - 2.1, name, 6.2, "right")
            positions.append(
                dict(city=row.city, eta=float(row.eta), y=yy, length_pt=length)
            )
    c.setStrokeColor(HexColor("#222222"))
    c.setLineWidth(0.65)
    c.rect(x, y, width, height, stroke=1, fill=0)
    for value in [0, 0.1, 0.2]:
        xx = x + value / maximum * width
        c.line(xx, y, xx, y + 2.5)
        txt(c, xx, y - 10, f"{value:.1f}", 6.1)
    txt(c, x + width / 2, y - 22, "η", 7)
    txt(c, x + width / 2, y + height + 6, "Top 5 / Bottom 5", 7.5, bold=True)
    c.setFillColor(HexColor("#555555"))
    for offset in [-4, 0, 4]:
        c.circle(x + width / 2, y + height / 2 + offset, 0.7, fill=1, stroke=0)
    return dict(
        axis_rect=B_RECT,
        xlim=[0, maximum],
        omitted_ranks=[6, 95],
        omitted_count=90,
        eta_axis_break=False,
        bar_positions=positions,
    )


def validation(c):
    x, y, w, h = C_RECT
    tx = lambda i: x + 5 + i * (w - 10) / 3
    ty = lambda eta: y + (eta - 0.10) / (0.22 - 0.10) * h
    axes(c, x, y, w, h, 0.10, 0.22, [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22])
    for values, color in [(EMPIRICAL, "#482173"), (SIMULATED, "#E76F51")]:
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(0.9)
        path = c.beginPath()
        path.moveTo(tx(0), ty(values[0]))
        for i, value in enumerate(values[1:], 1):
            path.lineTo(tx(i), ty(value))
        c.drawPath(path)
        c.setFillColor(HexColor("#FFFFFF"))
        for i, value in enumerate(values):
            c.circle(tx(i), ty(value), 1.55, stroke=1, fill=1)
    for i, name in enumerate(CITY_LABELS):
        c.setStrokeColor(HexColor("#222222"))
        c.line(tx(i), y, tx(i), y - 2.5)
        txt(c, tx(i), y - 10, name, 6)
    txt(c, x + w / 2, y - 23, "City", 7)
    for name, color, yy in [
        ("Empirical", "#482173", y + 15),
        ("Simulation", "#E76F51", y + 6),
    ]:
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(0.9)
        c.line(x + 5, yy, x + 14, yy)
        c.setFillColor(HexColor("#FFFFFF"))
        c.circle(x + 9.5, yy, 1.4, fill=1, stroke=1)
        txt(c, x + 17, yy - 2, name, 6, "left")


def boxes(c, groups, jitter):
    x, y, w, h = D_RECT
    gap = w / 4
    ty = lambda eta: y + (eta - 0.05) / (0.265 - 0.05) * h
    axes(c, x, y, w, h, 0.05, 0.265, [0.05, 0.10, 0.15, 0.20, 0.25])
    summaries = []
    for i, group in enumerate(groups):
        xx = x + (i + 0.5) * gap
        values = group.eta.to_numpy(float)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        lower = float(values[values >= q1 - 1.5 * iqr].min())
        upper = float(values[values <= q3 + 1.5 * iqr].max())
        half = 0.15 * gap
        c.setStrokeColor(HexColor("#536876"))
        c.setLineWidth(0.65)
        c.line(xx, ty(lower), xx, ty(q1))
        c.line(xx, ty(q3), xx, ty(upper))
        for end in [lower, upper]:
            c.line(xx - 0.55 * half, ty(end), xx + 0.55 * half, ty(end))
        c.saveState()
        c.setFillColor(HexColor(COLORS[i]))
        c.setFillAlpha(0.55)
        c.rect(xx - half, ty(q1), half * 2, ty(q3) - ty(q1), fill=1, stroke=1)
        c.restoreState()
        c.saveState()
        c.setFillColor(HexColor("#243F56"))
        c.setFillAlpha(0.38)
        for row in group.itertuples():
            c.circle(xx + jitter[row.city] * gap, ty(row.eta), 0.55, fill=1, stroke=0)
        c.restoreState()
        c.setStrokeColor(HexColor("#263D4C"))
        c.setLineWidth(1)
        c.line(xx - half, ty(med), xx + half, ty(med))
        c.setLineWidth(0.6)
        c.line(xx, y, xx, y - 2.5)
        txt(c, xx, y - 10, ["<0.5", "0.5–1", "1–5", "≥5"][i], 6.3)
        txt(c, xx, y + h + 6, f"n={len(group)}", 6)
        summaries.append(
            dict(
                group=i + 1,
                n=len(group),
                q1=float(q1),
                median=float(med),
                q3=float(q3),
                whisker_low=lower,
                whisker_high=upper,
            )
        )
    txt(c, x + w / 2, y - 25, "Population (million)", 7)
    return summaries


def draw_city_map(
    c,
    cells,
    normalized,
    bounds,
    cx,
    rotate_clockwise=False,
    max_width=92.0,
    top_offset=0.0,
):
    if rotate_clockwise:
        # Clockwise rotation maps east to down and north to right.
        cells = [
            ([np.column_stack((ring[:, 1], -ring[:, 0])) for ring in rings], value)
            for rings, value in cells
        ]
        xy = np.concatenate([ring for rings, value in cells for ring in rings])
        bounds = [*xy.min(axis=0), *xy.max(axis=0)]
    left, bottom, right, top = bounds
    # Fit within the width and height limits while preserving the aspect ratio.
    width = min(max_width, MAP_HEIGHT * (right - left) / (top - bottom))
    height = width * (top - bottom) / (right - left)
    x, y = cx - width / 2, MAP_TOP - height + top_offset
    scale = width / (right - left)
    c.saveState()
    for (rings, value), value_relative in zip(cells, normalized):
        c.setFillColor(rd_bu_r(float(value_relative**0.4), 0, 1))
        c.setFillAlpha(0.7)
        p = c.beginPath()
        for ring in rings:
            p.moveTo(x + (ring[0, 0] - left) * scale, y + (ring[0, 1] - bottom) * scale)
            for px, py in ring[1:]:
                p.lineTo(x + (px - left) * scale, y + (py - bottom) * scale)
            p.close()
        c.drawPath(p, stroke=0, fill=1, fillMode=1)
    c.restoreState()
    return [x, y, width, height]


def north_arrow(c, cx, clockwise, offset=53.0, y_offset=0.0):
    # Dedicated right gutter keeps arrows clear of all colored grid cells.
    x, y = cx + offset, 119.0 + y_offset
    direction = np.array([1.0, 0.0]) if clockwise else np.array([0.0, 1.0])
    normal = np.array([-direction[1], direction[0]])
    tail = np.array([x, y]) - direction * 5
    tip = np.array([x, y]) + direction * 5
    base = tip - direction * 3
    c.saveState()
    c.setStrokeColor(HexColor("#333333"))
    c.setFillColor(HexColor("#333333"))
    c.setLineWidth(0.65)
    c.line(*tail, *tip)
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*(base + normal * 1.6))
    p.lineTo(*(base - normal * 1.6))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    if clockwise:
        txt(c, x, y + 7, "N", 5.5)
    else:
        txt(c, x, y + 8, "N", 5.5)
    c.restoreState()
    return dict(
        direction="right" if clockwise else "up", tail=tail.tolist(), tip=tip.tolist()
    )


def colorbar(c, row_bottom):
    h = 124.0 - row_bottom
    top = 138.0 + pdfmetrics.getAscent("Arial-Bold", 6.7)
    x, y, w = 479.0, top - h, 5.5
    for i in range(256):
        c.setFillColor(rd_bu_r(i / 255, 0, 1))
        c.rect(x, y + i * h / 256, w, h / 256 + 0.01, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, fill=0, stroke=1)
    for v in [0, 0.5, 1]:
        yy = y + v**0.4 * h
        c.line(x + w, yy, x + w + 2, yy)
        txt(c, x + w + 4, yy - 2, f"{v:.1f}", 5.8, "left")
    c.saveState()
    c.translate(508, y + h / 2)
    c.rotate(90)
    txt(c, 0, 3, "Relative population", 5.7)
    txt(c, 0, -3.5, "per grid cell", 5.7)
    c.restoreState()
    return dict(rect=[x, y, w, h], gamma=0.4, label="Relative population per grid cell")


def marker(c, x, y, empirical, radius=1.7):
    c.setLineWidth(0.85)
    c.setStrokeColor(HexColor(PURPLE if empirical else ORANGE))
    c.setFillColor(white if empirical else HexColor(ORANGE))
    if empirical:
        c.circle(x, y, radius, stroke=1, fill=1)
    else:
        c.rect(
            x - radius * 0.82,
            y - radius * 0.82,
            radius * 1.64,
            radius * 1.64,
            stroke=1,
            fill=1,
        )


def paired(c, names, emp, sim):
    x, y, w, h = 392.0, 317.0, 115.0, 86.0
    ty = lambda v: y + (v - 0.10) / 0.12 * h
    tx = lambda i: x + 6 + i * (w - 12) / 3
    c.setStrokeColor(HexColor("#222222"))
    c.setLineWidth(0.65)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    for v in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]:
        yy = ty(v)
        c.line(x - 2.5, yy, x, yy)
        txt(c, x - 4, yy - 2, f"{v:.2f}", 6, "right")
    c.saveState()
    c.translate(x - 21, y + h / 2)
    c.rotate(90)
    txt(c, 0, 0, "η", 8)
    c.restoreState()
    for i, name in enumerate(names):
        xx = tx(i)
        c.setStrokeColor(HexColor("#9CA3A7"))
        c.setLineWidth(0.85)
        c.line(xx, ty(emp[i]), xx, ty(sim[i]))
        marker(c, xx, ty(emp[i]), True)
        marker(c, xx, ty(sim[i]), False)
        c.setStrokeColor(HexColor("#222222"))
        c.setLineWidth(0.6)
        c.line(xx, y, xx, y - 2.5)
        txt(c, xx, y - 10, name, 6)
    txt(c, x + w / 2, y - 23, "City", 7)
    for name, is_emp, yy in [("Empirical", True, y + 15), ("Simulation", False, y + 6)]:
        marker(c, x + 8, yy, is_emp, 1.5)
        txt(c, x + 15, yy - 2, name, 6, "left")


def arrow(c, x, y):
    c.setStrokeColor(HexColor("#333333"))
    c.setFillColor(HexColor("#333333"))
    c.setLineWidth(0.65)
    c.line(x, y - 5, x, y + 5)
    p = c.beginPath()
    p.moveTo(x, y + 5)
    p.lineTo(x - 1.6, y + 2)
    p.lineTo(x + 1.6, y + 2)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.setFillColor(HexColor("#222222"))
    c.setFont("Arial", 5.5)
    c.drawCentredString(x, y + 8, "N")


def run(data_root, output, dpi=600):
    data_root = Path(data_root)
    output = Path(output)
    pdfmetrics.registerFont(TTFont("Arial", publication_font(False)))
    pdfmetrics.registerFont(TTFont("Arial-Bold", publication_font(True)))
    data = pd.read_csv(data_root / "city_indicators.csv")[["city", "POP", "eta"]]
    assert len(data) == 407 and data.city.nunique() == 407 and data.city.notna().all()
    assert np.isfinite(data[["POP", "eta"]]).all().all() and (data.POP > 0).all()
    ranked = data.sort_values(["eta", "city"], ascending=[False, True]).reset_index(
        drop=True
    )
    ranked.insert(0, "eta_rank_all407", np.arange(1, 408))
    assert ranked.head(5).city.tolist() == [
        "fangchenggang",
        "dongguan",
        "beijing",
        "shanghai",
        "shenzhen",
    ]
    assert ranked.tail(5).city.tolist() == [
        "puning",
        "qiqihaer",
        "changji",
        "zhaodong",
        "shanwei",
    ]
    ids = pd.cut(data.POP, [0, 5e5, 1e6, 5e6, np.inf], labels=False, right=False) + 1
    groups = [data[ids == i] for i in range(1, 5)]
    assert [len(g) for g in groups] == [242, 88, 56, 21]
    rng = np.random.default_rng(20261015)
    jitter = dict(zip(sorted(data.city), rng.uniform(-0.11, 0.11, len(data))))
    layout = {"panel_a_rect": [4.0, 176.54782608695652, 224.0, 235.56521739130434]}
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf, pagesize=(W, H), pageCompression=1)
    c.setTitle("Figure 2")
    im = Image.open(data_root / "national_map.png").convert("RGB")
    a = im
    x, y, w, h = layout["panel_a_rect"]
    # Align the visible map content with the top of panels b and c.
    first_ink_row = int(
        np.flatnonzero((np.asarray(a).min(axis=2) < 230).sum(axis=1) >= 3)[0]
    )
    original_map_top = y + h - first_ink_row * w / a.width
    a_shift = 403.0 - original_map_top
    y += a_shift
    a_rect = [x, y, w, h]
    c.drawImage(ImageReader(a), x, y, width=w, height=h)
    panel(c, 8, 419, "a")
    panel(c, 221, 419, "b")
    ranking = merged_ranking(
        c,
        ranked.head(5),
        ranked.tail(5),
        data.eta.min(),
        data.eta.max(),
        float(ranked.eta.max()) * 1.06,
    )
    ranking.update(
        omitted_ranks=[6, 402], omitted_count=397, scope="all 407 study cities"
    )
    panel(c, 370, 419, "c")
    validation(c)
    panel(c, 370, 291, "d")
    stats = boxes(c, groups, jitter)
    panel(c, 8, 147, "e")
    maps = []
    for key, label, rank, path, cx in SPECS:
        row = ranked.iloc[rank - 1]
        assert row.city == key
        cells, norm, bounds, meta = population_map(key, path, data_root)
        clockwise = key == "fangchenggang"
        max_width = {"danzhou": 52.0, "yueyang": 54.0}.get(key, 92.0)
        offset_y = -8.0 if key == "danzhou" else 0.0
        rect = draw_city_map(c, cells, norm, bounds, cx, clockwise, max_width, offset_y)
        compass = north_arrow(
            c, cx, clockwise, 36.0 if key == "danzhou" else 53.0, offset_y
        )
        heading = f"{label} (#{rank}, η={row.eta:.3f})"
        txt(c, cx, 138, heading, 6.7, bold=True)
        meta.update(
            rank=rank,
            eta=float(row.eta),
            population=float(row.POP),
            heading=heading,
            pdf_rect=rect,
            xy_scale_equal=True,
            alpha=0.7,
            powernorm_gamma=0.4,
            heading_y=138,
            source_relative_path=path,
            rotation_clockwise_degrees=90 if clockwise else 0,
            north_arrow=compass,
            top_offset_pt=offset_y,
            normalization="(value-min)/(max-min) within city; finite positive cells",
        )
        maps.append(meta)
    row_bottom = min(m["pdf_rect"][1] for m in maps)
    bar = colorbar(c, row_bottom)
    c.save()

    # Draw the paired-point comparison and place its labels.
    pdf.seek(0)
    paired_buffer = io.BytesIO()
    overlay = canvas.Canvas(paired_buffer, pagesize=(W, H))
    overlay.setFillColor(white)
    for rect in [(366.0, 303.0, 152.0, 107.0), (389.0, 292.0, 129.0, 11.0)]:
        overlay.rect(*rect, fill=1, stroke=0)
    paired(overlay, CITY_LABELS, EMPIRICAL, SIMULATED)
    overlay.save()
    paired_buffer.seek(0)
    reader = PdfReader(pdf)
    reader.pages[0].merge_page(PdfReader(paired_buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    intermediate = io.BytesIO()
    writer.write(intermediate)
    intermediate.seek(0)
    spacing_buffer = io.BytesIO()
    overlay = canvas.Canvas(spacing_buffer, pagesize=(W, H))
    overlay.setFillColor(white)
    for rect in [
        (440.0, 291.0, 20.0, 11.0),
        (364.0, 112.0, 8.0, 21.0),
        (452.0, 104.0, 8.0, 23.0),
    ]:
        overlay.rect(*rect, fill=1, stroke=0)
    overlay.setFillColor(HexColor("#222222"))
    overlay.setFont("Arial", 7)
    overlay.drawCentredString(449.5, 298.0, "City")
    arrow(overlay, 351.0, 119.0)
    arrow(overlay, 456.0, 119.0)
    overlay.save()
    spacing_buffer.seek(0)
    reader = PdfReader(intermediate)
    reader.pages[0].merge_page(PdfReader(spacing_buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_metadata({"/Title": "Figure 2"})
    pdf_path = output / "Fig2.pdf"
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    subprocess.run(
        [
            str(publication_poppler()),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(output / "Fig2"),
        ],
        check=True,
    )
    with Image.open(output / "Fig2.png") as image:
        image.convert("RGB").resize(
            (1800, round(image.height * 1800 / image.width)), Image.Resampling.LANCZOS
        ).save(output / "Fig2_preview.png")
    ranked.to_csv(output / "eta_ranking.csv", index=False)
    pd.DataFrame(stats).to_csv(output / "population_group_statistics.csv", index=False)
    ranked[ranked.eta_rank_all407.isin([1, 250, 251, 400])].to_csv(
        output / "selected_cities.csv", index=False
    )
    summary = dict(
        cities=len(data),
        rank_order="descending eta; ascending city key for ties",
        validation_cities=CITY_LABELS,
        empirical_eta=EMPIRICAL,
        simulated_eta=SIMULATED,
        population_group_sizes=[len(g) for g in groups],
        map_cells=[m["retained_cells"] for m in maps],
    )
    (output / "analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
