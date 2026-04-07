#!/usr/bin/env python3
"""
laser_gen.py — Laser-Cut Text Art SVG Generator

Converts text passages into vector SVG art shaped as crosses, diamonds, hearts, etc.
Every glyph is converted to precise vector outlines using FreeType, then all paths
are boolean-unioned via Shapely into a single connected piece ready for laser cutting.

Usage:
    python laser_gen.py --text "HELLO WORLD" --font fonts/Impact.ttf --shape diamond
    python laser_gen.py --project mydesign.json --font fonts/Impact.ttf
    python laser_gen.py --text-file lyrics.txt --shape cross --auto-emphasis
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import freetype
from shapely.geometry import Polygon, MultiPolygon, box, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid

# ======================== CONSTANTS ========================

# Tier multipliers: 25:1 ratio (index = tier number, 0 unused)
TM = [0, 0.20, 0.35, 0.55, 1.00, 1.60, 2.80, 5.00]

# Auto-emphasis patterns
T7_RE = re.compile(r'^(HEAVEN|AMEN|FATHER|GOD|LORD|JESUS|CHRIST|EVIL|LOVE|ETERNAL|DREAD)$', re.I)
T6_RE = re.compile(r'^(HALLOWED|KINGDOM|FORGIVE|DELIVER|BREAD|EARTH|TRESPASSES?|TEMPTATION|POWER|NAME|COME|GLORY|SALVATION|PEACE|FAITH|HOPE|GRACE|DARK|SHADOWS?|SOULS?|OCEAN|FOREVER|MERCEDES|FRANCISCO|COUNTRY|STREETS|VALLEY|SHELL)$', re.I)
T5_RE = re.compile(r'^(WILL|DONE|GIVE|LEAD|DAILY|HOLY|SPIRIT|PRAYER|BIRTH|DYING|LYING|ENDING|SPACES|FACES|PILLS|DRINKS|THINK|HIDING|DRIVING|ARRIVING|WONDERING|CONCERNED)$', re.I)
T2_RE = re.compile(r'^(WHO|ART|BE|THY|ON|AS|IT|IS|NOT|BUT|FOR|AND|THE|A|AN|US|WE|IN|AT|TO|OF|BY|THIS|DAY|THOSE|AGAINST|FROM|THINE|EVER|INTO|WITH|THAT|OR|ITS|OUR|SO|IF|UP|OUT|OH|I|MY|SHE|HER|HIS|HIM)$', re.I)


# ======================== FONT / GLYPH VECTORIZATION ========================

def load_font(ttf_path: str) -> freetype.Face:
    face = freetype.Face(ttf_path)
    return face


def subdivide_quadratic(p0, p1, p2, n=10):
    """Subdivide a quadratic Bezier into line segments."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts


def subdivide_cubic(p0, p1, p2, p3, n=12):
    """Subdivide a cubic Bezier into line segments."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] +
             3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
        y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] +
             3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])
        pts.append((x, y))
    return pts


def outline_to_polygons(outline, scale, x_offset=0, y_offset=0):
    """Convert a FreeType outline to a list of Shapely Polygons.

    FreeType uses y-up, SVG uses y-down. We flip y here.
    Scale converts from font units to pixels.
    """
    points = [(p[0] * scale + x_offset, -p[1] * scale + y_offset)
              for p in outline.points]
    tags = outline.tags
    contour_ends = list(outline.contours)

    rings = []
    start = 0

    for end in contour_ends:
        contour_pts = points[start:end + 1]
        contour_tags = tags[start:end + 1]
        n = len(contour_pts)

        if n < 3:
            start = end + 1
            continue

        # Build line segments from the contour
        ring_pts = []
        i = 0
        while i < n:
            curr = contour_pts[i]
            curr_tag = contour_tags[i] & 1  # bit 0: on-curve

            if curr_tag:  # on-curve point
                ring_pts.append(curr)
                i += 1
            else:
                # Off-curve: need to handle conic (quadratic) beziers
                # TrueType: implied on-curve between consecutive off-curve
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n

                prev_pt = contour_pts[prev_idx]
                prev_on = contour_tags[prev_idx] & 1
                next_pt = contour_pts[next_idx]
                next_on = contour_tags[next_idx] & 1

                # Start point: previous on-curve or implied midpoint
                if prev_on:
                    p0 = prev_pt
                else:
                    p0 = ((prev_pt[0] + curr[0]) / 2, (prev_pt[1] + curr[1]) / 2)

                # End point: next on-curve or implied midpoint
                if next_on:
                    p2 = next_pt
                    i += 2  # skip next (it's the endpoint)
                else:
                    p2 = ((curr[0] + next_pt[0]) / 2, (curr[1] + next_pt[1]) / 2)
                    i += 1  # only skip control point, implied endpoint is next iteration's start

                # Subdivide the quadratic bezier
                bez_pts = subdivide_quadratic(p0, curr, p2, 8)
                ring_pts.extend(bez_pts[1:])  # skip first (already added as prev on-curve)

        # Close the ring
        if len(ring_pts) >= 3:
            ring_pts.append(ring_pts[0])
            rings.append(ring_pts)

        start = end + 1

    if not rings:
        return []

    # Determine exterior vs holes by signed area (winding direction)
    def signed_area(ring):
        a = 0
        for i in range(len(ring) - 1):
            a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
        return a / 2

    exteriors = []
    holes = []
    for ring in rings:
        area = signed_area(ring)
        if abs(area) < 0.1:
            continue
        if area > 0:  # clockwise in screen coords = exterior
            exteriors.append(ring)
        else:
            holes.append(ring)

    # Build polygons: assign holes to their containing exterior
    polys = []
    for ext in exteriors:
        try:
            p = Polygon(ext)
            if p.is_valid and p.area > 1:
                # Find holes that belong to this exterior
                my_holes = []
                for hole in holes:
                    try:
                        hp = Polygon(hole)
                        if p.contains(hp.representative_point()):
                            my_holes.append(hole)
                    except Exception:
                        pass
                if my_holes:
                    p = Polygon(ext, my_holes)
                if p.is_valid and p.area > 1:
                    polys.append(p)
        except Exception:
            pass

    # If no exteriors found, try treating all rings as exteriors (some fonts are weird)
    if not polys:
        for ring in rings:
            try:
                p = Polygon(ring)
                if not p.is_valid:
                    p = make_valid(p)
                if p.area > 1:
                    polys.append(p)
            except Exception:
                pass

    return polys


def text_to_polygons(face, text, font_size, x_offset=0, y_offset=0, letter_spacing_em=0):
    """Convert a string of text to Shapely polygons at the given position and size."""
    scale = font_size / face.units_per_EM
    face.set_char_size(int(font_size * 64))

    all_polys = []
    cursor_x = x_offset

    for i, char in enumerate(text):
        face.load_char(char, freetype.FT_LOAD_NO_BITMAP)
        outline = face.glyph.outline

        if outline.n_points > 0:
            polys = outline_to_polygons(outline, scale, cursor_x, y_offset)
            all_polys.extend(polys)

        # Advance cursor
        advance = face.glyph.advance.x / 64  # advance is in 26.6 fixed point
        cursor_x += advance + font_size * letter_spacing_em

    return all_polys


def measure_text(face, text, font_size, letter_spacing_em=0):
    """Measure text width, ascent, descent using FreeType metrics."""
    face.set_char_size(int(font_size * 64))

    width = 0
    for i, char in enumerate(text):
        face.load_char(char, freetype.FT_LOAD_NO_BITMAP)
        advance = face.glyph.advance.x / 64
        width += advance
        if i < len(text) - 1:
            width += font_size * letter_spacing_em

    scale = font_size / face.units_per_EM
    ascent = face.ascender * scale
    descent = -face.descender * scale  # FreeType descender is negative

    return width, ascent, descent


# ======================== SHAPE ROW FUNCTIONS ========================

def get_shape_row_fn(shape, W, H, params=None):
    """Return a function y -> (left, right) or None for the given shape."""
    params = params or {}
    P = 2  # minimal outer padding

    if shape == 'cross':
        arm_r = params.get('arm_width_pct', 32) / 100
        bar_r = params.get('bar_height_pct', 22) / 100
        bar_p = params.get('bar_position_pct', 30) / 100
        cx = W / 2
        arm_l = cx - W * arm_r / 2
        arm_rx = cx + W * arm_r / 2
        bar_top = H * bar_p - H * bar_r / 2
        bar_bot = H * bar_p + H * bar_r / 2

        def row_fn(y):
            if y < P or y > H - P:
                return None
            if bar_top <= y <= bar_bot:
                return (P, W - P)
            return (arm_l, arm_rx)
        return row_fn

    elif shape == 'diamond':
        cx, cy = W / 2, H / 2
        pts = [(cx, P), (W - P, cy), (cx, H - P), (P, cy)]
        return lambda y: _ray_poly(y, pts)

    elif shape == 'heart':
        cx, cy = W / 2, H / 2
        iw, ih = W - P * 2, H - P * 2
        pts = []
        for a_i in range(int(math.pi * 2 / 0.008)):
            a = a_i * 0.008
            x = 16 * math.sin(a) ** 3
            y_val = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
            pts.append((cx + x * iw / 36, cy + y_val * ih / 36 - ih * 0.06))
        return lambda y: _ray_poly(y, pts)

    elif shape == 'circle':
        cx, cy = W / 2, H / 2
        R = min(W, H) / 2 - P
        def row_fn(y):
            d = y - cy
            if abs(d) >= R:
                return None
            dx = math.sqrt(R * R - d * d)
            return (cx - dx, cx + dx)
        return row_fn

    elif shape == 'shield':
        cx, cy = W / 2, H / 2
        iw, ih = W - P * 2, H - P * 2
        my = cy + ih * 0.1
        pts = [(P, P), (W - P, P)]
        for i in range(41):
            t = i / 40
            pts.append((W - P - (W - P - cx) * t, my + (H - P - my) * t ** 0.8))
        for i in range(40, -1, -1):
            t = i / 40
            pts.append((P + (cx - P) * t, my + (H - P - my) * t ** 0.8))
        return lambda y: _ray_poly(y, pts)

    elif shape == 'oval':
        cx, cy = W / 2, H / 2
        rx = (W - P * 2) / 2
        ry = (H - P * 2) / 2 * 0.65
        def row_fn(y):
            d = (y - cy) / ry
            if abs(d) >= 1:
                return None
            dx = rx * math.sqrt(1 - d * d)
            return (cx - dx, cx + dx)
        return row_fn

    else:
        return get_shape_row_fn('diamond', W, H, params)


def _ray_poly(y, pts):
    """Find left/right x-intersections of horizontal line y with polygon."""
    xs = []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n if i < n - 1 else 0
        yi, yj = pts[i][1], pts[j][1]
        if (yi > y) != (yj > y):
            xi = pts[i][0] + (y - yi) / (yj - yi) * (pts[j][0] - pts[i][0])
            xs.append(xi)
    if len(xs) < 2:
        return None
    xs.sort()
    return (xs[0], xs[-1])


# ======================== LAYOUT ENGINE ========================

def build_tokens(words):
    """Group words into tokens (single words or stack groups)."""
    tokens = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.get('stackId'):
            group = []
            sid = w['stackId']
            start_idx = i
            while i < len(words) and words[i].get('stackId') == sid:
                group.append(words[i])
                i += 1
            tokens.append({'type': 'stack', 'words': group, 'idx': start_idx})
        else:
            tokens.append({'type': 'word', 'words': [w], 'idx': i})
            i += 1
    return tokens


def do_layout(words, row_fn, W, H, base_fs, face, vtight, ls_em, inner_pad):
    """Layout words into shape rows. Returns list of line dicts."""
    # Find vertical bounds
    top_y = None
    bot_y = None
    for y in range(int(H)):
        if row_fn(y) is not None:
            top_y = y
            break
    for y in range(int(H), 0, -1):
        if row_fn(y) is not None:
            bot_y = y
            break
    if top_y is None:
        return {'lines': [], 'all_fit': False, 'top_y': 0, 'bot_y': H, 'used_y': 0}

    tokens = build_tokens(words)
    lines = []
    cy = top_y + 1
    ti = 0

    while ti < len(tokens) and cy < bot_y:
        # Peek ahead to estimate tallest token
        peek_max = base_fs
        for p in range(ti, min(ti + 8, len(tokens))):
            tk = tokens[p]
            if tk['type'] == 'word':
                pfs = base_fs * TM[tk['words'][0]['tier']]
                if pfs > peek_max:
                    peek_max = pfs
            else:
                sh = sum(base_fs * TM[sw['tier']] for sw in tk['words'])
                if sh > peek_max:
                    peek_max = sh

        probe_y = cy + peek_max * 0.45
        row = row_fn(probe_y)
        if row is None or row[1] - row[0] < base_fs * 0.5:
            cy += 1
            continue

        avail_w = row[1] - row[0] - inner_pad * 2
        line_l = row[0] + inner_pad
        if avail_w < base_fs:
            cy += 1
            continue

        # Greedy pack tokens
        line_toks = []
        natural_w = 0
        tki = ti

        while tki < len(tokens):
            tk = tokens[tki]
            # Measure token natural width
            if tk['type'] == 'word':
                tok_fs = base_fs * TM[tk['words'][0]['tier']]
                tok_w, _, _ = measure_text(face, tk['words'][0]['text'], tok_fs, ls_em)
            else:
                n = len(tk['words'])
                est_fs = peek_max / n
                tok_w = 0
                for sw in tk['words']:
                    ww, _, _ = measure_text(face, sw['text'], est_fs, ls_em)
                    if ww > tok_w:
                        tok_w = ww

            gap = peek_max * max(0.05, 0.12 + ls_em) * 0.3 if line_toks else 0
            total = natural_w + gap + tok_w

            # Respect fit/break markers
            must_include = False
            if tki > ti:
                prev_tok = tokens[tki - 1]
                prev_last = prev_tok['words'][-1]
                if prev_last.get('lb') == 'fit':
                    must_include = True

            if total > avail_w and line_toks and not must_include:
                break

            line_toks.append({'tok': tk, 'nat_w': tok_w, 'gap': gap})
            natural_w = total
            tki += 1

            last_w = tk['words'][-1]
            if last_w.get('lb') == 'break':
                break

        if not line_toks:
            cy += 1
            continue

        # Scale factor to fill exact width
        scale = avail_w / natural_w if natural_w > 0 else 1

        # Compute maxH from non-stack tokens
        max_h = base_fs
        has_non_stack = False
        for lt in line_toks:
            if lt['tok']['type'] == 'word':
                fs = base_fs * TM[lt['tok']['words'][0]['tier']] * scale
                if fs > max_h:
                    max_h = fs
                has_non_stack = True
        if not has_non_stack:
            for lt in line_toks:
                sh = sum(base_fs * TM[sw['tier']] * scale for sw in lt['tok']['words'])
                if sh > max_h:
                    max_h = sh

        # Position tokens in internal (line-local) coordinates
        ix = 0
        positioned = []

        for j, lt in enumerate(line_toks):
            tk = lt['tok']
            gap = lt['gap'] * scale if j > 0 else 0
            ix += gap

            if tk['type'] == 'word':
                fs = base_fs * TM[tk['words'][0]['tier']] * scale
                ww, asc, desc = measure_text(face, tk['words'][0]['text'], fs, ls_em)
                positioned.append({
                    'text': tk['words'][0]['text'],
                    'x': ix,
                    'y': max_h * 0.82,  # baseline
                    'fs': fs,
                    'wi': tk['idx'],
                    'is_sep': tk['words'][0].get('isSep', False)
                })
                ix += ww
            else:
                # Stack
                slot_w = lt['nat_w'] * scale
                stack_words = []
                for si, sw in enumerate(tk['words']):
                    # Binary search for font size that fills slot_w
                    lo, hi, fs = 1, 300, 1
                    for _ in range(30):
                        mid = (lo + hi) / 2
                        mw, _, _ = measure_text(face, sw['text'], mid, ls_em)
                        if mw <= slot_w:
                            fs = mid
                            lo = mid + 0.01
                        else:
                            hi = mid - 0.01
                    m_w, m_asc, m_desc = measure_text(face, sw['text'], fs, ls_em)
                    stack_words.append({
                        'text': sw['text'], 'fs': fs, 'wi': tk['idx'] + si,
                        'asc': m_asc, 'desc': m_desc, 'mw': m_w
                    })

                # Pack using real ascent/descent
                sy = 0
                sub_words = []
                for sw in stack_words:
                    baseline = sy + sw['asc']
                    x_off = (slot_w - sw['mw']) / 2
                    sub_words.append({
                        'text': sw['text'], 'x': x_off, 'y': baseline,
                        'fs': sw['fs'], 'wi': sw['wi']
                    })
                    sy = baseline + sw['desc']

                view_box_h = sy
                positioned.append({
                    'type': 'stack',
                    'x': ix, 'w': slot_w,
                    'viewBoxH': view_box_h,
                    'words': sub_words
                })
                ix += slot_w

        lines.append({
            'cx': line_l, 'cy': cy, 'cw': avail_w, 'ch': max_h,
            'words': positioned,
            'h': max_h, 'top': cy, 'bot': cy + max_h
        })
        ti = tki
        cy += max_h * (vtight / 100)

    return {
        'lines': lines,
        'all_fit': ti >= len(tokens),
        'top_y': top_y,
        'bot_y': bot_y,
        'used_y': cy
    }


def find_best_layout(words, row_fn, W, H, face, vtight, ls_em, inner_pad):
    """Binary search for the largest base font size that fits all words."""
    lo, hi = 1, 80
    best_result = None
    best_base = 1

    for _ in range(50):
        mid = (lo + hi) / 2
        res = do_layout(words, row_fn, W, H, mid, face, vtight, ls_em, inner_pad)
        if res['all_fit']:
            best_base = mid
            best_result = res
            lo = mid + 0.02
        else:
            hi = mid - 0.02

    if not best_result:
        best_result = do_layout(words, row_fn, W, H, best_base, face, vtight, ls_em, inner_pad)

    # Vertical centering
    if best_result['lines']:
        first_top = best_result['lines'][0]['top']
        last_bot = best_result['lines'][-1]['bot']
        text_h = last_bot - first_top
        shape_h = best_result['bot_y'] - best_result['top_y']
        offset = (shape_h - text_h) / 2 - (first_top - best_result['top_y'])
        if offset > 1:
            for ln in best_result['lines']:
                ln['cy'] += offset

    return best_result, best_base


# ======================== VECTORIZATION & UNION ========================

def layout_to_polygons(layout, face, ls_em, bridge_width=2.0, fill_counters=False):
    """Convert the layout to Shapely polygons: text glyphs + bridges."""
    all_polys = []

    for ln in layout['lines']:
        lx, ly = ln['cx'], ln['cy']

        for w in ln['words']:
            if w.get('type') == 'stack':
                # Stack: scale internal coords to fit line height
                if w['viewBoxH'] > 0:
                    s = ln['ch'] / w['viewBoxH']
                else:
                    s = 1
                for sw in w['words']:
                    fs = sw['fs'] * s
                    x_pos = lx + w['x'] + sw['x'] * s
                    y_pos = ly + sw['y'] * s
                    polys = text_to_polygons(face, sw['text'], fs, x_pos, y_pos, ls_em)
                    if fill_counters:
                        polys = [Polygon(p.exterior) for p in polys if p.exterior is not None]
                    all_polys.extend(polys)
            elif w.get('is_sep'):
                continue  # skip separator symbols
            else:
                x_pos = lx + w['x']
                y_pos = ly + w['y']
                polys = text_to_polygons(face, w['text'], w['fs'], x_pos, y_pos, ls_em)
                if fill_counters:
                    polys = [Polygon(p.exterior) for p in polys if p.exterior is not None]
                all_polys.extend(polys)

    # Add bridges
    bridges = generate_bridges(layout, bridge_width)
    all_polys.extend(bridges)

    return all_polys


def generate_bridges(layout, bridge_width):
    """Generate connecting bridge rectangles between words and lines."""
    bridges = []
    lines = layout['lines']
    bw = bridge_width

    # Horizontal bridges between adjacent words on same line
    for ln in lines:
        words = [w for w in ln['words'] if not w.get('is_sep')]
        lx = ln['cx']
        for i in range(len(words) - 1):
            a = words[i]
            b = words[i + 1]

            if a.get('type') == 'stack':
                a_right = lx + a['x'] + a['w']
            else:
                a_right = lx + a['x'] + a.get('measured_w', a['fs'] * 2)

            if b.get('type') == 'stack':
                b_left = lx + b['x']
            else:
                b_left = lx + b['x']

            mid_y = ln['cy'] + ln['ch'] * 0.5
            left = min(a_right, b_left) - 1
            right = max(a_right, b_left) + 1
            bridges.append(box(left, mid_y - bw / 2, right, mid_y + bw / 2))

    # Vertical bridges between adjacent lines
    for i in range(len(lines) - 1):
        ln = lines[i]
        next_ln = lines[i + 1]
        top_y = ln['cy'] + ln['ch'] * 0.8
        bot_y = next_ln['cy'] + next_ln['ch'] * 0.2

        overlap_l = max(ln['cx'], next_ln['cx'])
        overlap_r = min(ln['cx'] + ln['cw'], next_ln['cx'] + next_ln['cw'])

        if overlap_r > overlap_l:
            span = overlap_r - overlap_l
            for frac in [0.15, 0.35, 0.5, 0.65, 0.85]:
                bx = overlap_l + span * frac
                bridges.append(box(bx - bw / 2, top_y, bx + bw / 2, bot_y))

    return bridges


def union_polygons(polys):
    """Union all polygons into a single geometry, handling invalid geometries."""
    # Fix invalid polygons first
    valid = []
    for p in polys:
        try:
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_valid:
                p = make_valid(p)
            if p.area > 0.5:
                valid.append(p)
        except Exception:
            pass

    if not valid:
        return None

    # Hierarchical union for performance
    # Union per batch of 50, then union the results
    while len(valid) > 1:
        batch_size = 50
        new_valid = []
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i + batch_size]
            try:
                merged = unary_union(batch)
                if merged.is_valid and not merged.is_empty:
                    new_valid.append(merged)
            except Exception:
                new_valid.extend(batch)
        valid = new_valid

    return valid[0] if valid else None


def ensure_connected(geom, bridge_width=3.0):
    """If geometry is MultiPolygon, add bridges to connect components."""
    if isinstance(geom, Polygon):
        return geom

    if not isinstance(geom, (MultiPolygon, GeometryCollection)):
        return geom

    parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    polys = [p for p in parts if isinstance(p, Polygon) and p.area > 1]

    if len(polys) <= 1:
        return polys[0] if polys else geom

    # Sort by area descending, connect smaller pieces to nearest larger
    polys.sort(key=lambda p: -p.area)
    bridges = []

    connected = [polys[0]]
    remaining = polys[1:]

    for _ in range(len(remaining)):
        if not remaining:
            break
        # Find the remaining polygon closest to any connected polygon
        best_dist = float('inf')
        best_ri = 0
        best_ci = 0
        for ri, rp in enumerate(remaining):
            for ci, cp in enumerate(connected):
                d = rp.distance(cp)
                if d < best_dist:
                    best_dist = d
                    best_ri = ri
                    best_ci = ci

        rp = remaining.pop(best_ri)
        cp = connected[best_ci]

        # Create bridge between nearest points
        rc = rp.centroid
        cc = cp.centroid
        bx = min(rc.x, cc.x) - bridge_width / 2
        by = min(rc.y, cc.y) - bridge_width / 2
        bw = abs(rc.x - cc.x) + bridge_width
        bh = abs(rc.y - cc.y) + bridge_width
        bridges.append(box(bx, by, bx + bw, by + bh))
        connected.append(rp)

    if bridges:
        all_parts = connected + bridges
        result = unary_union(all_parts)
        if result.is_valid:
            return result

    return unary_union(connected)


# ======================== SVG EXPORT ========================

def geometry_to_svg_path(geom, decimals=2):
    """Convert a Shapely geometry to an SVG path d attribute."""
    parts = []

    def ring_to_path(ring):
        coords = list(ring.coords)
        if not coords:
            return ''
        d = f'M{coords[0][0]:.{decimals}f},{coords[0][1]:.{decimals}f}'
        for x, y in coords[1:]:
            d += f' L{x:.{decimals}f},{y:.{decimals}f}'
        d += ' Z'
        return d

    if isinstance(geom, Polygon):
        parts.append(ring_to_path(geom.exterior))
        for hole in geom.interiors:
            parts.append(ring_to_path(hole))
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            if isinstance(g, Polygon):
                parts.append(ring_to_path(g.exterior))
                for hole in g.interiors:
                    parts.append(ring_to_path(hole))

    return ' '.join(parts)


def export_svg(geom, W, H, color='#000000', output_path='laser-output.svg'):
    """Write the geometry as a clean SVG file."""
    path_d = geometry_to_svg_path(geom)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <path d="{path_d}" fill="{color}" fill-rule="evenodd" stroke="none"/>
</svg>'''

    Path(output_path).write_text(svg, encoding='utf-8')
    return output_path


# ======================== AUTO EMPHASIS ========================

def auto_emphasis(words):
    """Apply automatic tier emphasis based on word significance."""
    for w in words:
        clean = re.sub(r'[^A-Za-z]', '', w['text'])
        if T7_RE.match(clean):
            w['tier'] = 7
        elif T6_RE.match(clean):
            w['tier'] = 6
        elif T5_RE.match(clean):
            w['tier'] = 5
        elif T2_RE.match(clean):
            w['tier'] = 2
        else:
            w['tier'] = 4
    return words


def parse_text(text, uppercase=True, separator=''):
    """Parse text into word list with optional separator tokens on newlines."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    words = []
    for li, line in enumerate(lines):
        for w in line.split():
            t = w.upper() if uppercase else w
            words.append({'text': t, 'tier': 4, 'lb': None, 'stackId': None, 'isSep': False})
        if separator and li < len(lines) - 1:
            if words:
                words[-1]['lb'] = 'break'
            words.append({'text': separator, 'tier': 3, 'lb': 'break', 'stackId': None, 'isSep': True})
    return words


def load_project(json_path):
    """Load a project JSON file saved by the JS tool."""
    with open(json_path) as f:
        data = json.load(f)
    words = []
    for w in data.get('WD', []):
        words.append({
            'text': w.get('text', ''),
            'tier': w.get('tier', 4),
            'lb': w.get('lb'),
            'stackId': w.get('stackId'),
            'isSep': w.get('isSep', False)
        })
    shape = data.get('curS', 'diamond')
    settings = data.get('settings', {})
    return words, shape, settings


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(description='Laser-Cut Text Art SVG Generator')
    parser.add_argument('--text', type=str, help='Input text')
    parser.add_argument('--text-file', type=str, help='Path to text file')
    parser.add_argument('--project', type=str, help='Path to JSON project file from JS tool')
    parser.add_argument('--font', type=str, default='C:/Windows/Fonts/impact.ttf', help='Path to .ttf font file')
    parser.add_argument('--shape', type=str, default='diamond', choices=['diamond', 'cross', 'heart', 'circle', 'shield', 'oval'])
    parser.add_argument('--width', type=int, default=600)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--bridge-width', type=float, default=2.0)
    parser.add_argument('--uppercase', action='store_true', default=True)
    parser.add_argument('--no-uppercase', action='store_false', dest='uppercase')
    parser.add_argument('--vtight', type=int, default=82, help='Vertical tightness 70-100')
    parser.add_argument('--letter-spacing', type=float, default=-0.02, help='Letter spacing in em')
    parser.add_argument('--padding', type=int, default=2, help='Inner padding')
    parser.add_argument('--arm-width', type=int, default=32, help='Cross arm width percent')
    parser.add_argument('--bar-height', type=int, default=22, help='Cross bar height percent')
    parser.add_argument('--bar-position', type=int, default=30, help='Cross bar position percent from top')
    parser.add_argument('--output', type=str, default='laser-output.svg')
    parser.add_argument('--fill-counters', action='store_true', help='Fill letter holes (O, A, D, etc)')
    parser.add_argument('--color', type=str, default='#000000')
    parser.add_argument('--auto-emphasis', action='store_true', help='Auto-assign word emphasis tiers')
    parser.add_argument('--separator', type=str, default='', help='Line separator symbol')
    parser.add_argument('--tiers', type=str, help='Comma-separated tier values per word')

    args = parser.parse_args()

    # Load font
    print(f'Loading font: {args.font}')
    face = load_font(args.font)

    # Get words
    if args.project:
        print(f'Loading project: {args.project}')
        words, shape, settings = load_project(args.project)
        if not args.shape:
            args.shape = shape
    elif args.text_file:
        text = Path(args.text_file).read_text(encoding='utf-8')
        words = parse_text(text, args.uppercase, args.separator)
    elif args.text:
        words = parse_text(args.text, args.uppercase, args.separator)
    else:
        parser.error('Must provide --text, --text-file, or --project')

    if args.auto_emphasis:
        words = auto_emphasis(words)

    if args.tiers:
        tiers = [int(t) for t in args.tiers.split(',')]
        for i, t in enumerate(tiers):
            if i < len(words):
                words[i]['tier'] = t

    print(f'{len(words)} words, shape={args.shape}, {args.width}x{args.height}')

    # Shape
    params = {
        'arm_width_pct': args.arm_width,
        'bar_height_pct': args.bar_height,
        'bar_position_pct': args.bar_position
    }
    row_fn = get_shape_row_fn(args.shape, args.width, args.height, params)

    # Layout
    print('Computing layout...')
    layout, base_fs = find_best_layout(
        words, row_fn, args.width, args.height,
        face, args.vtight, args.letter_spacing, args.padding
    )
    print(f'  Base font size: {base_fs:.1f}px, {len(layout["lines"])} lines')

    # Vectorize
    print('Converting text to vector outlines...')
    polys = layout_to_polygons(layout, face, args.letter_spacing, args.bridge_width, args.fill_counters)
    print(f'  {len(polys)} polygons generated')

    # Union
    print('Boolean union (this may take a moment)...')
    geom = union_polygons(polys)
    if geom is None:
        print('ERROR: No valid geometry produced')
        sys.exit(1)

    # Ensure single connected piece
    geom = ensure_connected(geom, args.bridge_width * 2)

    geom_type = type(geom).__name__
    if isinstance(geom, Polygon):
        print(f'  Result: single connected polygon, {len(list(geom.interiors))} holes')
    elif isinstance(geom, MultiPolygon):
        print(f'  Result: {len(list(geom.geoms))} separate pieces (bridges may need adjustment)')

    # Export
    output = export_svg(geom, args.width, args.height, args.color, args.output)
    print(f'Saved: {output}')
    print(f'  File size: {Path(output).stat().st_size / 1024:.1f} KB')


if __name__ == '__main__':
    main()
