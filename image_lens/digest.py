"""Short look.md: one line per tracking window, built from look.json. The detailed per-group text is look-full.md;
`focus` prints one shot of it. The whole-frame and whole-body shifts are said once per window, then only what the
limbs did on top of them, so one slide is not repeated for every tracked point. Bridges across a drawing swap are
pose differences between two frames: the path between them is not observed."""
import math
import re
from image_lens import body_motion, motion_report, onion, zoom

DIGEST = dict(parts=3, same_share=.5, repeat_px=8., repeat_share=.25)
ARM = ('손목', '팔꿈치', '엄지 끝', '검지 끝')
LEG = ('허벅지', '무릎', '발목')


def _dir(direction):
    return motion_report.KO[direction].replace('화면 ', '')


def _side(side):
    return '왼' if side == '화면 왼쪽' else '오른'


def _who(item, facing=None):
    """(word, screen) for a sided item: the dancer's own side from the pose model's joint name ('왼'/'오른') and the
    screen side it is on, from the window's facing (body_motion.facing) or, when that is unknown, where it was measured.
    Items without a model side (unnamed points, older records) keep the screen side alone. Screen-side names made a
    fresh reader of a front-facing dancer swap left and right in about half its limb statements."""
    body = item.get('body')
    if body not in ('l', 'r'):
        return _side(item['side']), None
    screen = item['side'] if facing is None else '화면 오른쪽' if (body == 'l') == (facing == 'front') else '화면 왼쪽'
    return ('왼' if body == 'l' else '오른'), screen


def _named(item, noun, facing=None):
    word, screen = _who(item, facing)
    return f"{word}{noun}" + (f"({screen})" if screen else '')


def arm_words(pose):
    """'왼팔 어깨 위·오른팔 허리 아래, 오른팔 몸 가운데 넘어감' for one body_motion.arm_poses entry; sides are the
    dancer's own (the window line says which way the dancer faces). '' when neither wrist was seen."""
    if not pose:
        return ''
    l, r = pose.get('l'), pose.get('r')
    if l and l == r:
        out = '양팔 ' + body_motion.ARM_KO[l]
    else:
        out = '·'.join(f"{word}팔 {body_motion.ARM_KO[h]}" for word, h in (('왼', l), ('오른', r)) if h)
    if not out:
        return ''
    cross = pose.get('cross') or []
    if len(cross) == 2:
        out += ', 두 팔 엇갈림'
    elif cross:
        out += f", {'왼' if cross[0] == 'l' else '오른'}팔 몸 가운데 넘어감"
    return out


def _speed(stroke):
    """Per-frame speed of a stroke; from the window's first frame its start is cut off (the window begins where the
    drawing changes, clip A f47: 40 px in two frames), so that is said too."""
    note = body_motion.timing_note(stroke)
    if note:
        return note
    text = motion_report.speed_text(stroke.get('speeds'))
    if not text:
        return ''
    return f"({text}" + (', 앞선 동작은 모름)' if stroke.get('from_start') else ')')


def limbs(body, facing=None):
    """Body-relative limb points merged into one statement per part: one screen-side arm or leg, or one unnamed move
    point. The lead is the point that travelled farthest (back-and-forth included), and only its strokes and timing are
    written: in the ease check a small-motion point set the label. Largest first."""
    groups = {}
    for limb in body.get('limbs', []):
        part = limb['part']
        kind = '팔' if part in ARM else '다리' if part in LEG else None
        word, screen = _who(limb, facing)
        key = (word + kind) if kind else part.replace('많이 움직인 곳', '움직인 점')
        groups.setdefault(key, dict(key=key, kind=kind, screen=screen, parts=[], limbs=[]))
        groups[key]['parts'].append(part); groups[key]['limbs'].append(limb)
    out = []
    for g in groups.values():
        lead = max(g['limbs'], key=lambda l: (l.get('travel_px', 0.), l['strokes'][0]['distance_px']))
        first = lead['strokes'][0]
        path = ''.join(('' if not i else ' → ' if s.get('run') == lead['strokes'][i-1].get('run') else ' / 관측 끊김 / ')
                       + f"{_dir(s['direction'])} {s['distance_px']:.0f}px{_speed(s)}" for i, s in enumerate(lead['strokes']))
        names = '·'.join(dict.fromkeys(p.replace(' 끝', '') for p in [lead['part']]+g['parts']))
        label = f"{g['key']}({g['screen']+': ' if g['screen'] else ''}{names})" if g['kind'] else g['key']
        axis = '' if not lead['axis_px'] else f", 몸 {'바깥' if lead['axis_px'] > 0 else '쪽'}으로"
        out.append(dict(key=g['key'], direction=first['direction'], distance=first['distance_px'], label=label,
                        point=lead['point_id'], text=f"{label} {path}{axis}"))
    return sorted(out, key=lambda g: -g['distance'])


def _screen(groups):
    """Screen moves from the group summaries (camera-following ones already removed), largest first. Points that moved
    together (same direction, time and size) are one move: the largest one speaks for them, with its own timing."""
    out = []
    for g in groups:
        moves = sorted(g.get('moves', []), key=lambda m: -m['distance_px']); taken = set()
        for t in g.get('together', []):
            members = [i for i, m in enumerate(moves) if i not in taken and m['point'] in t['points'] and m['direction'] == t['direction']
                       and m['frames'][0] <= t['frames'][1] and t['frames'][0] <= m['frames'][1]]
            if len(members) > 1:
                taken.update(members)
                out.append(dict(moves[members[0]], together=len(members)-1))
        out += [m for i, m in enumerate(moves) if i not in taken]
    return sorted(out, key=lambda m: -m['distance_px'])


def _bands(bands):
    """Per-band shifts, plus one sentence when bands go opposite ways sideways: in the blind test (clip A pose 1, head
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
    afresh per window, and on identical clip A drawings the pose model sometimes names the other arm."""
    out = []
    a, b = done.get('body'), first.get('body')
    if a and b:
        ca, cb = a['centre_shift'], b['centre_shift']
        moved = lambda c, body: c['distance_px'] >= body['limit_px']
        if ca is not None and (cb is None or moved(ca, a) != moved(cb, b) or (moved(ca, a) and (ca['direction'] != cb['direction'] or not _close(ca['distance_px'], cb['distance_px'])))):
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
            if not a.get('incomplete') and not any(same(m, o) for m in mine):
                out.append(f"{o['label']} {_dir(o['direction'])} {o['distance']:.0f}px 없음")
    ba, bb = done.get('bands') or [], first.get('bands') or []
    if len(ba) == len(bb) and any(x['dx'] is not None and y['dx'] is not None and math.dist((x['dx'], x['dy']), (y['dx'], y['dy']))
                                  >= max(DIGEST['repeat_px'], DIGEST['repeat_share']*max(math.hypot(x['dx'], x['dy']), math.hypot(y['dx'], y['dy'])))
                                  for x, y in zip(ba, bb)):
        out.append('높이별 ' + _bands(ba))
    return out


def camera(cam, fps):
    """The whole-frame motion as one phrase, or None when it did not move."""
    if not cam:
        return None
    least = motion_report.STROKE['major_min_px']
    net = cam.get('net')
    if 'known' not in cam:  # look.json from before camera.describe: the net shift only
        return f"카메라 {_dir(motion_report.direction(*net))} {math.hypot(*net):.0f}px" if net and math.hypot(*net) >= least else None
    if cam.get('unknown'):
        return f"카메라 모름({100*cam['known']:.0f}%만 맞춤)"
    parts = []
    if net and math.hypot(*net) >= least:
        parts.append(f"카메라 {_dir(motion_report.direction(*net))} {math.hypot(*net):.0f}px"
                     + (f"({motion_report.speed_text(cam['speeds'])})" if cam.get('speeds') else ''))
    if cam.get('back'):
        b = cam['back']
        parts.append(f"카메라 {_dir(motion_report.direction(b['dx'], b['dy']))} {math.hypot(b['dx'], b['dy']):.0f}px까지 갔다 돌아옴({b['frame']/fps:.2f}s)")
    if cam.get('zoom'):
        parts.append(('줌 인' if cam['zoom'] > 1 else '줌 아웃') + f" ×{cam['zoom']:.2f}")
    elif cam.get('zoom_back'):
        z = cam['zoom_back']
        parts.append(('줌 인' if z['scale'] > 1 else '줌 아웃') + f" ×{z['scale']:.2f}까지 갔다 돌아옴({z['frame']/fps:.2f}s)")
    if parts and cam.get('until'):
        parts[-1] += f" (f{cam['until']}까지만 맞춤)"
    return ', '.join(parts) or None


def clip_camera(record):
    """One header sentence on why the camera is unknown when no window placed it, else None. The windows keep their own
    "카메라 모름": leaving it out there scored 12.3 against 14.1 in six reading runs each (solo dance clip; p about .07)."""
    cams = [d['camera'] for s in record['shots'] for d in s['windows'] if d.get('camera')]
    if not cams or not all(c.get('unknown') for c in cams):
        return None
    lo, hi = min(c['known'] for c in cams), max(c['known'] for c in cams)
    share = f"{100*lo:.0f}%" if lo == hi else f"{100*lo:.0f}–{100*hi:.0f}%"
    why = '배경이 거의 민무늬라 맞출 점이 없음' if all(c.get('plain') for c in cams) else '배경이 프레임끼리 맞지 않음'
    return (f"카메라: 모든 구간에서 모름({why}, 구간마다 {share} 프레임만 맞춤). 고정인지 움직였는지는 이 도구로 알 수 없고, "
            '화면 이동 px에 카메라 움직임이 섞였을 수 있다.')


def window_line(done, fps, first=None):
    """One window. With `first` (the first window of the same repeated pose) only the differences are written."""
    w = done['window']; end = done.get('cut_after', w['end'])
    fields = [f"{w['seed']/fps:.2f}–{(end+1)/fps:.2f}s f{w['seed']}..{end}"]
    if done.get('framing'):
        fields.append(body_motion.FRAMING_KO[done['framing']] + (' (팔만)' if done.get('scope') == 'upper' else ''))
    if done.get('facing'):
        # Also where the text names no limbs (low skeleton trust): a reader looking at the pictures mirrored the dancer.
        fields.append('정면(사람 왼쪽=화면 오른쪽)' if done['facing'] == 'front' else '뒤돎(사람 왼쪽=화면 왼쪽)')
    # Arm posture at the start, and at the end unless the step lines below say it.
    words = [arm_words(a) or '모름' for a in done.get('arms') or []]
    if done.get('phases') or len(words) == 1:
        words = words[:1]
    if words and set(words) != {'모름'}:
        fields.append('팔 자세: ' + (f"{words[0]} 그대로" if len(words) == 2 and words[0] == words[1] else ' → '.join(words)))
    camera_text = camera(done.get('camera'), fps)
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
            whole.append('몸 관측 끊김·기준점 변경, 전체 이동 모름' if c is None else
                         f"몸 {_dir(c['direction'])} {c['distance_px']:.0f}px{_speed(c)}" if c['distance_px'] >= body['limit_px'] else '몸 그대로')
        if whole:
            fields.append('전체: ' + ', '.join(whole))
        if body:
            found = limbs(body, done.get('facing'))
            if found:
                shown = [f['text'] for f in found[:DIGEST['parts']]]
                fields.append('몸 기준: ' + '; '.join(shown) + (f" (+{len(found)-DIGEST['parts']})" if len(found) > DIGEST['parts'] else ''))
            elif not body.get('unknown') and not body.get('uncertain'):
                fields.append('몸 기준: 관측된 연속 구간에서 큰 이동 미검출(실제 동작은 모름)'
                              if body.get('incomplete') or ((done.get('body_geometry') or {}).get('source') in ('raw_tracker_source_switch', 'per_joint_source_stable') or (done.get('body_geometry') or {}).get('excluded_motion_frames'))
                              else f"몸 기준: 팔다리 {body['limit_px']:.0f}px 미만")
        else:
            moves = _screen(done.get('groups', []))
            if moves:
                shown = [f"{m['point']}" + (f" 외 {m['together']}점 함께" if m.get('together') else '')
                         + f" {_dir(m['direction'])} {m['distance_px']:.0f}px" + (f"({motion_report.speed_text(m['speeds'])})" if m.get('speeds') else '')
                         for m in moves[:DIGEST['parts']]]
                fields.append('화면: ' + '; '.join(shown) + (f" (+{len(moves)-DIGEST['parts']})" if len(moves) > DIGEST['parts'] else ''))
            elif (done.get('body_geometry') or {}).get('source') in ('raw_tracker_source_switch', 'per_joint_source_stable'):
                fields.append('몸 기준: 원시 기준점 부족, 움직임 모름')
            elif (done.get('body_geometry') or {}).get('excluded_motion_frames'):
                fields.append('몸 기준: 관절 추정 충돌 프레임 제외, 움직임 모름'
                              if (done.get('body_geometry') or {}).get('conflict_motion_frames')
                              else '몸 기준: 관절 돌출 교정 프레임 제외, 움직임 모름')
            elif done.get('groups'):
                fields.append('화면: 24px 이상 이동 없음')
            else:  # seed or every track failed: nothing was measured, which is not stillness
                fields.append('추적 결과 없음: 움직임 모름')
        flow = _bands(done.get('bands'))
        if flow:
            fields.append('높이별: ' + flow)
    fields += events(done.get('events'), fps, done.get('facing'))
    fields += people(done.get('people'), fps)
    notes = []
    geometry = done.get('body_geometry')
    note = body_motion.geometry_note(geometry)
    if note:
        notes.append(note)
        # First windows already say this in the motion field; repeat windows need it here too.
        if first is not None:
            body = done.get('body')
            if not body:
                notes.append('원시 기준점 부족, 몸 동작 모름')
            elif not body.get('limbs') and not body.get('unknown') and not body.get('uncertain'):
                notes.append('관측된 연속 구간에서 큰 이동 미검출(실제 동작은 모름)')
    body = done.get('body')
    if body and body.get('incomplete'):
        notes.append('일부 관측 끊김·짧은 관측: 그 사이 동작 모름')
    if first is not None and body and body.get('centre_shift') is None:
        notes.append('몸 전체 이동 모름')
    if done.get('facing') == 'back':
        notes.append('뒤돈 자세(엉덩이로 판단): 팔다리 왼/오른은 모델 추정, 틀리기 쉬움')
    if done.get('chibi'):
        notes.append('SD 체형: 팔다리 이름 안 붙임')
    elif not done.get('skeleton', True):
        notes.append('골격 신뢰 낮음')
    hidden = [_named(u, u['kind'], done.get('facing')) for u in (done.get('body') or {}).get('unknown', [])]
    if hidden:
        notes.append('·'.join(hidden) + ' 대부분 가려져 모름')
    unsure = [_named(u, u['part'], done.get('facing')) for u in (done.get('body') or {}).get('uncertain', [])]
    if unsure:
        notes.append('·'.join(unsure) + ' 불확실(자세·추적 어긋남)')
    lost = [g for g in done.get('groups', []) if g.get('lost')]
    if lost:
        notes.append('끊김 ' + ', '.join(f"{g['group']} f{g['lost']['after']}~" for g in lost))
    if done.get('cut_after') is not None:
        notes.append(f"f{done['cut_after']} 뒤 " + ('점 끊김(그림 그대로)' if done.get('cut_redrawn') is False else '그림 바뀜'))
    warnings = sum(g.get('warnings', 0) for g in done.get('groups', []))
    if warnings:
        notes.append(f'경고 {warnings}')
    if notes:
        fields.append('비고: ' + ', '.join(notes))
    return '- ' + ' | '.join(fields)


def phase_lines(done, fps):
    """Indented lines under a window: its 1-2 s steps (body_motion.phases), so a reader sees the moves the window's
    first-to-last line and onion lose."""
    out = []; steps = done.get('phases') or []
    ends = {e['frame']: e for e in (done.get('arms') or [])[1:]}
    for p in steps:
        a, b = p['frames']; b += p is steps[-1]  # the last step ends after its frame, like the window
        said = [f"{_named(m, body_motion.PHASE_KO[m['part']], done.get('facing'))} {_dir(m['direction'])} {m['amount']:.1f}"
                + (f"(최대 {m['back']:.1f} 갔다 돌아옴)" if m['back'] else '') for m in p['moved'][:body_motion.PHASES['parts']]]
        pose = arm_words(ends.get(p['frames'][1]))
        out.append(f"  - {a/fps:.2f}–{b/fps:.2f}s: " + ('; '.join(said) if said else '팔다리 거의 그대로')
                   + (f" | 끝 팔: {pose}" if pose else ''))
    return out

def events(found, fps, facing=None):
    """Fields from the per-frame pose: hand near the face, each shoulder's rises and falls, shoulder-width changes, and
    hands mostly unseen. A limb below the stroke threshold is not still when its hand was out of view (clip B 13 s:
    hands spreading off the bottom edge read as "정지")."""
    if not found:
        return []
    out = []
    near = [f"{_named(e, '손', facing)}이 얼굴({e['region']}) " + ('가까이 옴' if e.get('enters', True) else '처음부터 가까이')
            + f" {e['frames'][0]/fps:.2f}–{(e['frames'][1]+1)/fps:.2f}s" + (', 떨어짐' if e.get('exits', True) else ', 끝까지') for e in found['hand_face']]
    if near:
        out.append('손-얼굴: ' + '; '.join(near[:DIGEST['parts']]) + (f" (+{len(near)-DIGEST['parts']})" if len(near) > DIGEST['parts'] else ''))
    touch = [(f"{_named(e, '손', facing)}이 {_named(dict(body='r' if e['body'] == 'l' else 'l', side=e['to_side']), '어깨', facing)}"
              if e['to'] == 'shoulder' else '두 손') + (' 가까이 옴' if e['to'] == 'shoulder' else ' 모음')
             + ('' if e.get('enters', True) else '(처음부터)') + f" {e['frames'][0]/fps:.2f}–{(e['frames'][1]+1)/fps:.2f}s"
             + (', 떨어짐' if e.get('exits', True) else ', 끝까지') for e in found.get('hand_body', [])]
    if touch:
        out.append('손-몸: ' + '; '.join(touch[:DIGEST['parts']]) + (f" (+{len(touch)-DIGEST['parts']})" if len(touch) > DIGEST['parts'] else ''))
    moves = {}
    for e in found['shoulders']:
        moves.setdefault(_named(e, '어깨', facing), []).append(('올라감' if e['rise'] else '내려감') + f" {e['amount']:.2f}({e['frames'][0]/fps:.2f}s)")
    if moves:
        out.append('어깨: ' + '; '.join(f"{name} " + ' → '.join(m[:DIGEST['parts']]) for name, m in sorted(moves.items())))
    if found.get('turns'):
        out.append('몸 돌림 후보(어깨 폭 크게 변함, 포즈가 튄 것일 수도; 그동안 어깨 높이는 안 씀): '
                   + ', '.join(f"{a/fps:.2f}–{(b+1)/fps:.2f}s" for a, b in found['turns'][:DIGEST['parts']]))
    elif found['width']:
        out.append('어깨 폭: ' + ' → '.join(f"{'넓어짐' if e['wider'] else '좁아짐'} {100*e['share']:.0f}%({e['frames'][0]/fps:.2f}s)"
                                           for e in found['width'][:DIGEST['parts']]))
    if found.get('lost_frames', 0) >= max(3, round(.1*fps)):
        out.append(f"처음 사람의 포즈 추적이 {found['lost_frames']}프레임 끊김: 그동안 모름")
    if 'hands_body' in found:
        hidden = [_named(dict(body=b), '손', facing) if facing else ('왼' if b == 'l' else '오른')+'손'
                  for b, share in sorted(found['hands_body'].items()) if share < body_motion.EVENTS['hands_seen']]
    else:
        hidden = [_side(side)+'손' for side, share in sorted(found['hands_seen'].items()) if share < body_motion.EVENTS['hands_seen']]
    if hidden:
        out.append(f"{'·'.join(hidden)} 대부분 안 보임: 멈춤 아님, 모름")
    return out


def people(found, fps):
    """Fields for windows with several people: who is where, how each moved, falls, and pair relations."""
    if not found or len(found['people']) < 2:
        return []
    each = []
    for p in found['people']:
        text = f"{p['name']} " + (f"{_dir(p['move']['direction'])} {p['move']['widths']:.1f}폭" if p['move']['widths'] >= body_motion.PEOPLE['move'] else '거의 그대로')
        if p.get('fall'):
            text += f", 쓰러짐 후보({p['fall']['frame']/fps:.2f}s 몸통 {p['fall']['degrees']}°)"
        if p['seen'] < 1:
            text += f", {100*p['seen']:.0f}%만 보임"
        each.append(text)
    out = [f"사람 {len(found['people'])}명(P1부터 화면 왼쪽, 몸 중심 이동은 어깨 폭 단위, 카메라 뺌): " + '; '.join(each)]
    for r in found['relations']:
        a, b = r['pair']; parts = []
        if r.get('distance'):
            parts.append(('다가감' if r['distance']['closer'] else '멀어짐') + f" {r['distance']['widths']:.1f}폭")
        parts += [f"{t['hand']} 손이 {t['body']} 어깨·몸 쪽 {t['frames'][0]/fps:.2f}–{(t['frames'][1]+1)/fps:.2f}s(닿음 후보)" for t in r['touch'][:DIGEST['parts']]]
        if r['overlap']:
            parts.append('상자가 많이 겹침: 포즈가 섞였을 수 있음')
        if parts:
            out.append(f"{a}–{b}: " + ', '.join(parts))
    return out


def bridge_line(bridge, facing=None):
    if not bridge or not bridge['limbs']:
        return None
    a, b = bridge['frames']
    # The largest change on each screen side: by size alone both could be one arm, and the other hand's 294 px drop
    # (clip A f61->f62) was lost in the blind test.
    largest = {}
    for l in sorted(bridge['limbs'], key=lambda l: -l['distance_px']):
        largest.setdefault(l['side'], l)
    shown = sorted(largest.values(), key=lambda l: -l['distance_px'])
    return (f"- 전환 f{a}→f{b} (앞뒤 자세 차이, 사이 경로 모름): "
            + '; '.join(f"{_named(l, l['part'], facing)} {_dir(l['direction'])} {l['distance_px']:.0f}px" for l in shown))


def markdown(record, scan, full_name):
    fps = scan['fps']; total = scan['source_frame_count']
    covered = sum(min(d.get('cut_after', d['window']['end']), d['window']['end'])-d['window']['seed']+1
                  for s in record['shots'] for d in s['windows'] if d.get('groups'))
    skipped = [k for s in record['shots'] for k in s['skipped']]
    out = ['# 영상 움직임 요약', '',
           f"영상 {record['source_video'].rsplit('/', 1)[-1]}: {total}프레임, {fps:g}fps, 장면 {len(record['shots'])}개. "
           f"추적된 구간 {covered}/{total}프레임, 건너뜀 {len(skipped)}곳.",
           '읽는 법: 한 줄 = 추적 구간 하나. 방향은 화면 기준. 팔·다리·손·어깨의 왼/오른은 사람 자신의 몸 기준(포즈 모델의 관절 이름; 앞을 보면 왼팔이 화면 오른쪽), '
           '괄호 안은 그 구간에서 화면 어느 쪽인지(엉덩이 방향으로 앞·뒤를 판단). 이름 없는 점(move-a…)과 높이별·사람 순서(P1…)는 화면 기준. "전체"는 카메라와 몸 중심(어깨 가운데, 없으면 얼굴)의 이동이고, '
           '"몸 기준"은 그 위에 팔다리가 더 한 움직임(몸 중심 기준, 같은 쪽 팔·다리에서 같이 움직인 점은 한 줄). '
           '"(속도 9→5→1 px/f)"는 그 이동을 시간 순서로 나눈 프레임당 속도(앞이 크면 느려짐, 뒤가 크면 빨라짐; 몸 기준은 몸 중심을 뺀 속도). '
           '"높이별"은 사람 상자를 위→아래 5칸으로 나눈 이동(px, 0=8px 미만). 골격 신뢰가 낮으면 팔다리 이름 없이 "화면" 이동만. '
           '"자세 A"는 처음과 끝 그림이 같은 구간이 다시 나온 것: 반복은 처음과 다른 점만 쓴다. '
           '"카메라 모름"이면 화면 이동에 카메라 움직임이 섞였을 수 있다. '
           '어깨는 엉덩이(없으면 눈) 기준 높이 변화, 숫자는 어깨 폭 단위. 어깨 폭은 눈 사이 거리 기준: 넓어짐=가슴 폄·카메라 쪽으로 돎, 좁아짐=웅크림·옆으로 돎. 시간 다음 칸은 구도(전신/무릎 위/상반신/얼굴 클로즈업, 포즈 관절이 화면 안에 있는지로 판단): 상반신·클로즈업이면 사람 상자가 화면 아래에서 잘려서 '
           '높이별 아래 칸은 다리가 아니라 가슴·손이나 얼굴 아래쪽이고, "(팔만)"은 엉덩이 없이 어깨로 팔만 이름 붙였다는 뜻. '
           '들여 쓴 "- 1.20–2.80s:" 줄은 2초 넘는 구간을 가장 멈춘 프레임에서 1~2초 단계로 나눈 것: 단계 처음→끝 팔다리 끝점의 몸 기준 이동(어깨 가운데 기준, 숫자는 몸통 길이 단위: 어깨~엉덩이), "갔다 돌아옴"은 단계 안에서 멀리 갔다가 돌아온 점(괄호 안이 가장 멀리 간 거리). '
           '"팔 자세"(구간 처음→끝)·"끝 팔"(단계 끝)은 그 순간 손목 높이(어깨 위/어깨 높이/가슴 높이/허리 아래)와 손목이 몸 가운데선을 넘었는지. '
           '"손-몸"은 손이 반대 어깨 가까이 오거나 두 손이 모인 구간(화면 위 거리라 닿음 후보).',
           f"자세히: `image_lens focus --look <이 폴더> --shot <shot-id>` (또는 {full_name}). 점 이름·부위는 미검증, 수치는 원본 비교 전 후보.", '']
    if record.get('joint_tracking'):
        out.insert(3, '몸 관절 좌표는 pose·추적 혼합 추정(보임·관절 정답 미검증). 속도·완급은 연속된 원시 점 추적에서만 계산. *.raw-tracks.json은 원시 결과, *.tracks.json은 선택된 좌표. SD·골격 불명은 이름 없는 표면점 경로.')
    if record.get('scaled'):
        out.insert(3, f"원본 {record['scaled']['from'][0]}×{record['scaled']['from'][1]}을 {record['scaled']['to'][0]}×{record['scaled']['to'][1]}로 줄여서 봄: px는 줄인 영상 기준.")
    if record.get('onion_sheets') or record.get('frame_sheets'):
        # Same 30 facts of a solo dance, fresh reader: frames only 6, this text with onion sheets 8, with every sheet 12.5.
        out.insert(-1, '그림과 같이 읽기: 글만으로는 중간 동작·손 모양·닿음이 빠진다. 아래 그림을 모두 열어 글과 같이 본다. 가장 믿을 만한 근거는 궤적이 그려진 어니언 그림이다(실제 장면 위에 추적한 점을 프레임마다 그대로 찍은 것이고, 글은 그것을 줄인 것). 글과 그림이 어긋나면 그림을 믿는다. 그다음 strip, 전체 화면 모음(시간 순서).')
    if record.get('onion_sheets'):
        out.insert(-1, f"어니언 모음 그림: {', '.join(record['onion_sheets'])} (구간마다 청록=시작, 빨강=끝, 초록 화살표=높이별 이동, 오른쪽 회색=시작과 가장 다른 가운데 프레임: 돌거나 갔다 돌아오는 동작은 여기서 보임; "
                       f"색 점선=각 줄 \"몸 기준\"의 부위 대표 점 궤적({'·'.join(onion.TRAILS['names'])} = 몸 기준에 적힌 순서, 점 하나가 한 프레임, 카메라 뺌, 큰 원=시작·큰 네모=끝, 점 간격이 좁으면 느림); "
                       '"who"=사람이 여럿인 구간의 P1·P2 상자; "still"=추적하기엔 짧은 컷의 한 장면, 붙어 있는 컷은 한 줄에 시간 순서; "strip"=몸 돌림 후보 구간의 프레임 5장을 시간 순서로). '
                       '손 모양(하트·브이)처럼 글에 안 나오는 동작은 그림에서 본다.')
    if record.get('frame_sheets'):
        out.insert(-1, f"전체 화면 모음: {', '.join(record['frame_sheets'])} (전 구간 균등 추출, 최대 {zoom.ZOOM['max_overview']}장, 장마다 시간 표시; 사이에 빠진 빠른 동작은 여기 안 나옴: "
                       '사람끼리 하는 동작(당기기·기대기·찌르기)과 넘어짐은 포즈 상자가 합쳐지면 글에 안 나오거나 엉뚱한 사람에게 붙으니 여기서 본다.)')
    whole_camera = clip_camera(record)
    if whole_camera:
        out.insert(3, whole_camera)
    from image_lens import check
    out += check.lines(record.get('check'))
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
            out.append(f"- 건너뜀 f{k['frames'][0]}..{k['frames'][1]}: {k['reason']}" + (' (모음 그림에 still 사진: 자세만, 움직임 모름)' if k.get('still') else ''))
        for done in s['windows']:
            bridge = done.get('bridge'); pair = (prev.get('motif'), done.get('motif')) if prev else (None, None)
            if bridge and bridge['limbs'] and all(pair) and pair in switches:
                a, b = bridge['frames']; was = switches[pair]
                out.append(f"- 전환 f{a}→f{b}: 자세 {pair[0]}→{pair[1]}, f{was[0]}→f{was[1]}과 같은 자세 변화")
            else:
                line = bridge_line(bridge, done.get('facing'))
                if line:
                    out.append(line)
                if bridge and bridge['limbs'] and all(pair):
                    switches[pair] = bridge['frames']
            label = done.get('motif')
            out.append(window_line(done, fps, first.get(label)))
            out += phase_lines(done, fps)
            if label:
                first.setdefault(label, done)
            prev = done
        if s['errors']:
            out.append(f"- 오류 {len(s['errors'])}건 (look.json)")
    if record.get('errors'):
        out.append(f"전체 오류 {len(record['errors'])}건: " + ', '.join(e['step'] for e in record['errors']) + ' (look.json)')
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
