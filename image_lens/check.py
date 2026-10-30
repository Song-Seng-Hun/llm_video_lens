"""Where the text is most likely wrong, so the reader looks there first. A reader does not pick the spans it got wrong
(clip D 18-22 s, a pull and a fall in one merged pose box, was written confidently as an arm around the shoulder and
not chosen for zoom), so the tool ranks its own warning signs and shows those spans as dense frames. Since pose.py
splits merged boxes, that window reads as two people with a fall candidate, but it still scores 6 and is not in the
top four, so frame-sheet-N.jpg stays the guard against it.
`at` gives everything the look has on one span the reader asks about."""
from image_lens import zoom

# Weight per sign. Merged boxes, nothing tracked and frames with no pose at all say the text may be missing the event;
# a low skeleton score, a turn candidate or an unknown camera say only that the numbers are rough.
CHECK = dict(spans=4, min_score=2, overlap=3, untracked=3, no_pose=3, lost=2, warnings=2, pose_lost=2, short_cuts=2,
             short_run=3, no_pose_frames=12, people=1, skeleton=1, turns=1, camera=1, merge_gap_s=.5)


def _end(done):
    return done.get('cut_after', done['window']['end'])


def signs(record, fps):
    """[(first frame, last frame, score, [reasons], anchor frames)] from look.json, one per window or skipped range."""
    found = []; shorts = []
    for s in record['shots']:
        for k in s['skipped']:
            a, b = k['frames']
            if k['reason'] == 'no unflagged pose track' and b-a+1 >= CHECK['no_pose_frames']:
                found.append((a, b, CHECK['no_pose'], ['포즈 없어 글에 없음'], []))
            elif k['reason'] != 'no unflagged pose track':  # too short to track: a still at most
                shorts.append((a, b))
        for d in s['windows']:
            a, b = d['window']['seed'], _end(d); score = 0; why = []; anchors = []
            def add(weight, text, frames=()):
                nonlocal score
                score += weight; why.append(text); anchors.extend(frames)
            people = (d.get('people') or {}).get('people', [])
            if any(r['overlap'] for r in (d.get('people') or {}).get('relations', [])):
                add(CHECK['overlap'], '사람 상자 겹침: 포즈 섞였을 수 있음')
            elif len(people) > 1:
                add(CHECK['people'], f'사람 {len(people)}명')
            lost = [g for g in d.get('groups', []) if g.get('lost')]
            if lost:
                after = sorted({g['lost']['after'] for g in lost})
                add(CHECK['lost'], '점 끊김 ' + ', '.join(f'f{n}~' for n in after), after)
            warnings = sum(g.get('warnings', 0) for g in d.get('groups', []))
            if warnings:
                add(CHECK['warnings'], f'경고 {warnings}')
            events = d.get('events') or {}
            if events.get('lost_frames', 0) >= max(3, round(.1*fps)):
                add(CHECK['pose_lost'], f"포즈 추적 {events['lost_frames']}프레임 끊김")
            if d.get('skeleton') is False:
                add(CHECK['skeleton'], 'SD 체형' if d.get('chibi') else '골격 신뢰 낮음')
            if events.get('turns'):
                add(CHECK['turns'], f"몸 돌림 후보 {len(events['turns'])}곳", [t[0] for t in events['turns']])
            if (d.get('camera') or {}).get('unknown'):
                add(CHECK['camera'], '카메라 모름')
            if score:
                found.append((a, b, score, why, anchors))
    # Back-to-back short cuts (the clip B spin at 1.8-2.2 s was seven cuts of 3-6 frames) are only stills in the text.
    run = []
    for a, b in sorted(shorts)+[(None, None)]:
        if a is not None and run and a-run[-1][1] <= CHECK['merge_gap_s']*fps:
            run.append((a, b)); continue
        if len(run) >= CHECK['short_run']:
            found.append((run[0][0], run[-1][1], CHECK['short_cuts'], [f'짧은 컷 {len(run)}개: 사진만, 움직임 모름'], []))
        run = [(a, b)] if a is not None else []
    return found


def _clip(a, b, anchors, fps):
    """At most zoom.ZOOM['span_s'] seconds of [a, b], placed on the most anchors."""
    room = int(zoom.ZOOM['span_s']*fps)-1
    if b-a <= room:
        return a, b
    best = max(range(a, b-room+1), key=lambda x: (sum(x <= f <= x+room for f in anchors), -x))
    return best, best+room


def spans(record, fps):
    """The CHECK['spans'] highest-scoring windows or ranges at CHECK['min_score'] or more, each cut to at most
    zoom.ZOOM['span_s'], in time order. Windows touch each other, so signs are ranked one by one, not merged."""
    out = []
    for a, b, score, why, anchors in sorted(signs(record, fps), key=lambda m: (-m[2], m[0])):
        if score < CHECK['min_score'] or len(out) == CHECK['spans']:
            break
        first, last = _clip(a, b, anchors, fps)
        out.append(dict(frames=[first, last], seconds=[round(first/fps, 2), round((last+1)/fps, 2)], score=score, reasons=why))
    return sorted(out, key=lambda c: c['frames'])


def sheets(video, root, found):
    """check-sheet-N.jpg, one per span, sharing zoom's frame budget; adds `image` to each span."""
    import cv2
    if not found:
        return []
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        each = max(4, min(zoom.ZOOM['frames'], zoom.ZOOM['budget']//len(found)))
        written = zoom._write(cap, fps, [(i, zoom.picks(*c['frames'], each)) for i, c in enumerate(found)], root, 'check-sheet')
    finally:
        cap.release()
    for w in written:
        found[w['span']]['image'] = w['image']
    return [w['image'] for w in written]


def lines(found):
    if not found:
        return []
    return (['## 먼저 확인할 곳', '도구가 스스로 불안하다고 본 구간이다(점수 순이 아니라 시간 순). 여기 없는 곳도 틀릴 수 있다. '
             '자세한 근거: `image_lens at --look <이 폴더> --span <시작-끝초>`.']
            + [f"- {c['seconds'][0]:.2f}–{c['seconds'][1]:.2f}s" + (f" ({c['image']})" if c.get('image') else '') + ': ' + ', '.join(c['reasons'])
               for c in found] + [''])


def at(look_dir, span, out):
    """Everything the look has on one span (start-end seconds, at most zoom.ZOOM['span_s']): the window lines, skipped
    ranges, transitions and check signs that overlap it, the pictures to open, and dense frames of the span.
    Writes <out>/at.md and <out>/at-frames-1.jpg."""
    import cv2
    from image_lens import digest, paths
    (a_s, b_s), = zoom.spans([span])
    look = paths.confined(f'{look_dir}/look.json', exists=True)
    record, _ = paths.load(look, max_bytes=64*1024*1024)
    scan, _ = paths.load(look.with_name('shots.json'))
    fps = scan['fps']; total = scan['source_frame_count']
    a, b = round(a_s*fps), min(round(b_s*fps), total-1)
    if a > total-1:
        raise ValueError(f'span starts after the video ends ({total/fps:.2f} s)')
    video = paths.confined(record['source_video'], exists=True, max_bytes=512*1024*1024)
    root = paths.confined(out, new=True)
    def overlaps(first, last):
        return first <= b and last >= a
    body = []; pictures = []
    for s in record['shots']:
        if not overlaps(*s['frames']):
            continue
        body.append(f"## {s['segment_id']} {s['seconds'][0]:.2f}–{s['seconds'][1]:.2f}s")
        for k in s['skipped']:
            if overlaps(*k['frames']):
                body.append(f"- 건너뜀 f{k['frames'][0]}..{k['frames'][1]}: {k['reason']}" + (f" (사진 {k['still']}: 자세만)" if k.get('still') else ''))
                if k.get('still'):
                    pictures.append(k['still'])
        for d in s['windows']:
            if not overlaps(d['window']['seed'], _end(d)):
                continue
            line = digest.bridge_line(d.get('bridge'))
            if line:
                body.append(line)
            body.append(digest.window_line(d, fps))
            pictures += [d[k] for k in ('onion', 'who') if d.get(k)] + ([d['strip']['image']] if d.get('strip') else [])
    warned = [f"- {c['seconds'][0]:.2f}–{c['seconds'][1]:.2f}s: {', '.join(c['reasons'])}" for c in spans(record, fps) if overlaps(*c['frames'])]
    cap = cv2.VideoCapture(str(video))
    try:
        root.mkdir(parents=True)
        frames = zoom._write(cap, fps, [(span, zoom.picks(a, b, zoom.ZOOM['frames']))], root, 'at-frames')
    finally:
        cap.release()
    text = ([f'# {a/fps:.2f}–{(b+1)/fps:.2f}s (f{a}..{b})', '',
             '이 구간에 걸친 look 결과만 모았다. 구간 줄은 걸친 부분만이 아니라 구간 전체의 요약이다. 수치는 원본 비교 전 후보.',
             f"촘촘한 프레임: {', '.join(f['image'] for f in frames)}" + (f"; 구간 그림(look 폴더): {', '.join(dict.fromkeys(pictures))}" if pictures else ''), '']
            + (['## 도구가 불안하다고 본 곳'] + warned + [''] if warned else [])
            + (body or ['이 구간에 걸친 장면 없음']))
    md = paths.write(root/'at.md', '\n'.join(text).rstrip('\n')+'\n')
    return dict(status='written', output=str(md), frames=frames[0]['frames'] if frames else [],
                cost=dict(images=len(frames), pixels=sum(f['pixels'] for f in frames)))
