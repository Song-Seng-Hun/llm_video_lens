"""Short look.md: one line per tracking window, built from look.json. The detailed per-group text is look-full.md;
`focus` prints one shot of it. The whole-frame and whole-body shifts are said once per window, then only what the
limbs did on top of them, so one slide is not repeated for every tracked point. Bridges across a drawing swap are
pose differences between two frames: the path between them is not observed."""
import math
import re
from image_lens import motion_report

DIGEST = dict(parts=3, same_share=.5, repeat_px=8., repeat_share=.25)
ARM = ('손목', '팔꿈치', '엄지 끝', '검지 끝')
LEG = ('허벅지', '무릎', '발목')


def _dir(direction):
    return motion_report.KO[direction].replace('화면 ', '')


def _side(side):
    return '왼' if side == '화면 왼쪽' else '오른'


def limbs(body, ease=None):
    """Body-relative limb points merged into statements: points of one screen-side arm or leg whose first stroke has
    the same direction and a distance within half of each other are one statement. Largest first."""
    ease = ease or {}
    groups = []
    for limb in body.get('limbs', []):
        part = limb['part']; first = limb['strokes'][0]
        kind = '팔' if part in ARM else '다리' if part in LEG else None
        key = (_side(limb['side']) + kind) if kind else part.replace('많이 움직인 곳', '움직인 점')
        for g in groups:
            if g['key'] == key and g['direction'] == first['direction'] \
                    and min(g['distance'], first['distance_px']) >= DIGEST['same_share']*max(g['distance'], first['distance_px']):
                g['parts'].append(part); g['distance'] = max(g['distance'], first['distance_px']); g['limbs'].append(limb)
                break
        else:
            groups.append(dict(key=key, kind=kind, direction=first['direction'], distance=first['distance_px'], parts=[part], limbs=[limb]))
    out = []
    for g in sorted(groups, key=lambda g: -g['distance']):
        lead = max(g['limbs'], key=lambda l: l['strokes'][0]['distance_px'])
        path = ' → '.join(f"{_dir(s['direction'])} {s['distance_px']:.0f}px" for s in lead['strokes'])
        names = '·'.join(dict.fromkeys(p.replace(' 끝', '') for p in g['parts']))
        label = f"{g['key']}({names})" if g['kind'] else g['key']
        axis = '' if not lead['axis_px'] else f", 몸 {'바깥' if lead['axis_px'] > 0 else '쪽'}으로"
        timing = f", {motion_report.EASE_KO[ease[lead['point_id']]]}" if ease.get(lead['point_id']) else ''
        out.append(dict(key=g['key'], direction=g['direction'], distance=g['distance'], label=label, text=f"{label} {path}{axis}{timing}"))
    return out


def _screen(groups):
    """Screen moves from the group summaries (camera-following ones already removed), largest first."""
    moves = [m for g in groups for m in g.get('moves', [])]
    return sorted(moves, key=lambda m: -m['distance_px'])


def _bands(bands):
    """Per-band shifts, plus one sentence when bands go opposite ways sideways: in the blind test (AKAGE pose 1, head
    right 22 / waist left 18-29) two readers given only the numbers did not read them as "the waist goes left"."""
    if not bands or all(b['dx'] is None for b in bands):
        return None
    parts = []; sides = {}
    for b in bands:
        if b['dx'] is None:
            parts.append('?')
        else:
            d = math.hypot(b['dx'], b['dy']); direction = motion_report.direction(b['dx'], b['dy'])
            parts.append('0' if d < 8 else f"{_dir(direction)} {d:.0f}")
            side = 'right' if 'right' in direction else 'left' if 'left' in direction else None
            if d >= 8 and side:
                sides.setdefault('오른쪽' if side == 'right' else '왼쪽', []).append(b['band'])
    text = ' / '.join(parts)
    if len(sides) == 2:
        text += ' (' + ', '.join(f"{'·'.join(names)} {side}" for side, names in sides.items()) + ': 높이에 따라 좌우가 반대)'
    return text


def _close(a, b):
    return abs(a-b) < max(DIGEST['repeat_px'], DIGEST['repeat_share']*max(a, b))


def _differences(done, first):
    """What a repeat did differently from the pose's first showing: body-centre shift, limb statements and height bands.
    Statements match by direction and size (8 px or a quarter) whatever their label: move-point letters are assigned
    afresh per window, and on identical AKAGE drawings the pose model sometimes names the other arm."""
    out = []
    a, b = done.get('body'), first.get('body')
    if a and b:
        ca, cb = a['centre_shift'], b['centre_shift']
        moved = lambda c, body: c['distance_px'] >= body['limit_px']
        if moved(ca, a) != moved(cb, b) or (moved(ca, a) and (ca['direction'] != cb['direction'] or not _close(ca['distance_px'], cb['distance_px']))):
            out.append(f"몸 {_dir(ca['direction'])} {ca['distance_px']:.0f}px" if moved(ca, a) else '몸 그대로')
        mine, theirs = limbs(a), limbs(b)
        turn = lambda m, o: min((motion_report.DIRECTIONS.index(m['direction'])-motion_report.DIRECTIONS.index(o['direction'])) % 8,
                                (motion_report.DIRECTIONS.index(o['direction'])-motion_report.DIRECTIONS.index(m['direction'])) % 8)
        # Neighbouring directions count as the same: 26 px "up" against 25 px "up-left" read as a missing rise in the blind test.
        same = lambda m, o: turn(m, o) <= 1 and _close(m['distance'], o['distance'])
        for m in mine:
            if not any(same(m, o) for o in theirs):
                out.append(m['text'])
        for o in theirs:
            if not any(same(m, o) for m in mine):
                out.append(f"{o['label']} {_dir(o['direction'])} {o['distance']:.0f}px 없음")
    ba, bb = done.get('bands') or [], first.get('bands') or []
    if len(ba) == len(bb) and any(x['dx'] is not None and y['dx'] is not None and math.dist((x['dx'], x['dy']), (y['dx'], y['dy']))
                                  >= max(DIGEST['repeat_px'], DIGEST['repeat_share']*max(math.hypot(x['dx'], x['dy']), math.hypot(y['dx'], y['dy'])))
                                  for x, y in zip(ba, bb)):
        out.append('높이별 ' + _bands(ba))
    return out


def window_line(done, fps, first=None):
    """One window. With `first` (the first window of the same repeated pose) only the differences are written."""
    w = done['window']; end = done.get('cut_after', w['end'])
    fields = [f"{w['seed']/fps:.2f}–{(end+1)/fps:.2f}s f{w['seed']}..{end}"]
    cam = (done.get('camera') or {}).get('net')
    camera_text = f"카메라 {_dir(motion_report.direction(*cam))} {math.hypot(*cam):.0f}px" \
        if cam and math.hypot(*cam) >= motion_report.STROKE['major_min_px'] else None
    if first is not None:
        fw = first['window']
        diff = _differences(done, first)
        fields.append(f"자세 {done['motif']} 반복 ({fw['seed']/fps:.2f}s와 " + ('같음' if not diff else '같고, 다른 점: ' + '; '.join(diff[:DIGEST['parts']])
                      + (f" (+{len(diff)-DIGEST['parts']})" if len(diff) > DIGEST['parts'] else '')) + ')')
        if camera_text:
            fields.append('전체: ' + camera_text)
    else:
        if done.get('motif'):
            fields.append(f"자세 {done['motif']}")
        whole = [camera_text] if camera_text else []
        body = done.get('body')
        if body:
            c = body['centre_shift']
            whole.append(f"몸 {_dir(c['direction'])} {c['distance_px']:.0f}px" if c['distance_px'] >= body['limit_px'] else '몸 그대로')
        if whole:
            fields.append('전체: ' + ', '.join(whole))
        ease = {m['point']: m['ease'] for m in sorted(_screen(done.get('groups', [])), key=lambda m: m['distance_px']) if m.get('ease')}
        if body:
            found = limbs(body, ease)
            if found:
                shown = [f['text'] for f in found[:DIGEST['parts']]]
                fields.append('몸 기준: ' + '; '.join(shown) + (f" (+{len(found)-DIGEST['parts']})" if len(found) > DIGEST['parts'] else ''))
            else:
                fields.append(f"몸 기준: 팔다리 {body['limit_px']:.0f}px 미만")
        else:
            moves = _screen(done.get('groups', []))
            if moves:
                shown = [f"{m['point']} {_dir(m['direction'])} {m['distance_px']:.0f}px" + (f", {motion_report.EASE_KO[m['ease']]}" if m.get('ease') else '')
                         for m in moves[:DIGEST['parts']]]
                fields.append('화면: ' + '; '.join(shown) + (f" (+{len(moves)-DIGEST['parts']})" if len(moves) > DIGEST['parts'] else ''))
            else:
                fields.append('화면: 24px 이상 이동 없음')
        flow = _bands(done.get('bands'))
        if flow:
            fields.append('높이별: ' + flow)
    notes = []
    if not done.get('skeleton', True):
        notes.append('골격 신뢰 낮음')
    lost = [g for g in done.get('groups', []) if g.get('lost')]
    if lost:
        notes.append('끊김 ' + ', '.join(f"{g['group']} f{g['lost']['after']}~" for g in lost))
    if done.get('cut_after') is not None:
        notes.append(f"f{done['cut_after']} 뒤 그림 바뀜")
    warnings = sum(g.get('warnings', 0) for g in done.get('groups', []))
    if warnings:
        notes.append(f'경고 {warnings}')
    if notes:
        fields.append('비고: ' + ', '.join(notes))
    return '- ' + ' | '.join(fields)


def bridge_line(bridge):
    if not bridge or not bridge['limbs']:
        return None
    a, b = bridge['frames']
    # The largest change on each screen side: by size alone both could be one arm, and the other hand's 294 px drop
    # (AKAGE f61->f62) was lost in the blind test.
    largest = {}
    for l in sorted(bridge['limbs'], key=lambda l: -l['distance_px']):
        largest.setdefault(l['side'], l)
    shown = sorted(largest.values(), key=lambda l: -l['distance_px'])
    return (f"- 전환 f{a}→f{b} (앞뒤 자세 차이, 사이 경로 모름): "
            + '; '.join(f"{_side(l['side'])}{l['part']} {_dir(l['direction'])} {l['distance_px']:.0f}px" for l in shown))


def markdown(record, scan, full_name):
    fps = scan['fps']; total = scan['source_frame_count']
    covered = sum(min(d.get('cut_after', d['window']['end']), d['window']['end'])-d['window']['seed']+1
                  for s in record['shots'] for d in s['windows'])
    skipped = [k for s in record['shots'] for k in s['skipped']]
    out = ['# 영상 움직임 요약', '',
           f"영상 {record['source_video'].rsplit('/', 1)[-1]}: {total}프레임, {fps:g}fps, 장면 {len(record['shots'])}개. "
           f"추적 구간 {covered}/{total}프레임, 건너뜀 {len(skipped)}곳.",
           '읽는 법: 한 줄 = 추적 구간 하나. 방향·좌우는 화면 기준. "전체"는 카메라와 몸 중심(어깨 가운데, 없으면 얼굴)의 이동이고, '
           '"몸 기준"은 그 위에 팔다리가 더 한 움직임(몸 중심 기준, 같은 쪽 팔·다리에서 같이 움직인 점은 한 줄). '
           '"높이별"은 사람 상자를 위→아래 5칸으로 나눈 이동(px, 0=8px 미만). 골격 신뢰가 낮으면 팔다리 이름 없이 "화면" 이동만. '
           '"자세 A"는 처음과 끝 그림이 같은 구간이 다시 나온 것: 반복은 처음과 다른 점만 쓴다.',
           f"자세히: `image_lens focus --look <이 폴더> --shot <shot-id>` (또는 {full_name}). 점 이름·부위는 미검증, 수치는 원본 비교 전 후보.", '']
    windows = [d for s in record['shots'] for d in s['windows']]
    seen = {}
    for d in windows:
        if d.get('motif'):
            seen.setdefault(d['motif'], []).append(d)
    if seen:
        out += ['## 반복 자세'] + [f"- 자세 {k}: {len(v)}번 (" + ', '.join(f"{d['window']['seed']/fps:.2f}s" for d in v) + ')' for k, v in seen.items()] + ['']
    first = {}; prev = None; switches = {}
    for s in record['shots']:
        out.append(f"## {s['segment_id']} {s['seconds'][0]:.2f}–{s['seconds'][1]:.2f}s")
        for k in s['skipped']:
            out.append(f"- 건너뜀 f{k['frames'][0]}..{k['frames'][1]}: {k['reason']}")
        for done in s['windows']:
            bridge = done.get('bridge'); pair = (prev.get('motif'), done.get('motif')) if prev else (None, None)
            if bridge and bridge['limbs'] and all(pair) and pair in switches:
                a, b = bridge['frames']; was = switches[pair]
                out.append(f"- 전환 f{a}→f{b}: 자세 {pair[0]}→{pair[1]}, f{was[0]}→f{was[1]}과 같은 자세 변화")
            else:
                line = bridge_line(bridge)
                if line:
                    out.append(line)
                if bridge and bridge['limbs'] and all(pair):
                    switches[pair] = bridge['frames']
            label = done.get('motif')
            out.append(window_line(done, fps, first.get(label)))
            if label:
                first.setdefault(label, done)
            prev = done
        if s['errors']:
            out.append(f"- 오류 {len(s['errors'])}건 (look.json)")
    return '\n'.join(out).rstrip('\n')+'\n'


def focus(look_dir, shot):
    """The section of look-full.md for one shot."""
    from image_lens import paths
    if not re.fullmatch(r'shot-\d{4}', shot):
        raise ValueError('shot must look like shot-0001')
    text = paths.confined(f'{look_dir}/look-full.md', exists=True).read_text(encoding='utf-8')
    for section in text.split('\n## '):
        if section.startswith(shot+' '):
            return '## '+section.rstrip('\n')+'\n'
    raise ValueError(f'{shot} not in look-full.md')
