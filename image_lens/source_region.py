"""Explicit native source crop; downstream geometry remains whole-analysis-source geometry."""
import copy
import hashlib
import math
import os
import re
import time
from image_lens import paths

MAX_BYTES = 512*1024*1024


def parse_box(value):
    if isinstance(value, str):
        if not re.fullmatch(r'[0-9]+,[0-9]+,[0-9]+,[0-9]+', value):
            raise ValueError('--scene-box requires integer X0,Y0,X1,Y1')
        value = [int(v) for v in value.split(',')]
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError('--scene-box requires four integer native coordinates')
    x0,y0,x1,y1 = value
    if min(x0,y0) < 0 or x1 <= x0 or y1 <= y0 or (x1-x0)%2 or (y1-y0)%2:
        raise ValueError('--scene-box must be nonempty, nonnegative and have even width/height')
    return list(value)


def _mapping(box, size):
    x0,y0,x1,y1 = box; sx,sy = (x1-x0)/size[0],(y1-y0)/size[1]
    return dict(scope='raw_analysis_only', convention='pixel_centres: parent=(analysis+.5)*crop/size-.5+origin',
                matrix3x3=[[sx,0.,x0+(sx-1)/2],[0.,sy,y0+(sy-1)/2],[0.,0.,1.]])


def header(region):
    return (f"요청 영역 {','.join(map(str,region['box']))}만 분석함: 픽셀 좌표는 분석한 잘라낸 영상"
            f"({region['analysis']['width']}×{region['analysis']['height']}, 크기 조정 시 조정본) 기준이며, 원본 대응은 source_region의 raw 좌표 변환만 사용한다.")


def prepare(source, root, box, *, max_pixels=1920*1080, max_seconds=1800):
    import cv2
    box = parse_box(box); source = paths.confined(source, exists=True, max_bytes=MAX_BYTES)
    final = paths.confined(root/'source-region.avi', new=True)
    partial = paths.confined(root/'source-region.partial.avi', new=True)
    deadline = time.monotonic()+max_seconds
    def bounded():
        if time.monotonic() >= deadline:raise ValueError('source-region conversion exceeded time budget')
        if partial.exists() and partial.stat().st_size > MAX_BYTES:raise ValueError('source-region copy exceeds byte limit')
    parent_hash = paths.sha(source); parent_bytes = source.stat().st_size
    cap = cv2.VideoCapture(str(source)); writer = None
    try:
        w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (not cap.isOpened() or cap.getBackendName() != 'FFMPEG' or not hasattr(cv2,'CAP_PROP_PTS')
                or not math.isfinite(fps) or fps <= 0 or min(w,h) <= 0 or count < 2):
            raise ValueError('source-region requires known FFmpeg CFR timestamps and frame geometry')
        if box[2] > w or box[3] > h:raise ValueError('--scene-box outside original input frame')
        cw,ch = box[2]-box[0],box[3]-box[1]; size = (cw,ch)
        if cw*ch > max_pixels:
            k = math.sqrt(max_pixels/(cw*ch)); size = (int(cw*k)//2*2,int(ch*k)//2*2)
        if min(size) < 2:raise ValueError('source-region resized dimensions too small')
        writer = cv2.VideoWriter(str(partial),cv2.VideoWriter_fourcc(*'FFV1'),fps,size)
        if not writer.isOpened():raise ValueError('lossless FFV1 writer unavailable; no lossy fallback')
        hashes = []; start_pts = start_msec = None
        while True:
            bounded(); ok,image = cap.read()
            if not ok:break
            if image.shape[:2] != (h,w):raise ValueError('source-region input frame geometry changed')
            pts,msec = cap.get(cv2.CAP_PROP_PTS),cap.get(cv2.CAP_PROP_POS_MSEC)
            if start_pts is None:start_pts,start_msec = pts,msec
            n = len(hashes)
            if (not all(math.isfinite(v) for v in (pts,msec)) or abs(pts-start_pts-n)>1e-3
                    or abs((msec-start_msec)*fps/1000-n)>1e-3):
                raise ValueError('VFR or unknown timestamps: source-region requires validated CFR')
            image = image[box[1]:box[3],box[0]:box[2]]
            if size != (cw,ch):image = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
            hashes.append(hashlib.sha256(image.tobytes()).digest());writer.write(image)
        if len(hashes) != count:raise ValueError('source-region decoded frame count differs from input metadata')
        writer.release();writer = None;cap.release();bounded()
        check = cv2.VideoCapture(str(partial))
        try:
            if not check.isOpened():raise ValueError('cannot verify lossless source-region copy')
            if abs(check.get(cv2.CAP_PROP_FPS)-fps)>1e-6:
                raise ValueError('lossless writer cannot preserve source frame rate')
            for n,expected in enumerate(hashes):
                bounded();ok,image = check.read()
                if (not ok or image.shape[:2] != size[::-1] or hashlib.sha256(image.tobytes()).digest() != expected
                        or abs(check.get(cv2.CAP_PROP_PTS)-n)>1e-3
                        or abs(check.get(cv2.CAP_PROP_POS_MSEC)*fps/1000-n)>1e-3):
                    raise ValueError('lossless source-region pixel/frame/timestamp verification failed')
            if check.read()[0]:raise ValueError('source-region copy has extra frames')
        finally:check.release()
        if paths.sha(source) != parent_hash or source.stat().st_size != parent_bytes:
            raise ValueError('parent source changed during source-region conversion')
        metadata = dict(schema='image-lens.source-region/1',
            parent=dict(video=str(source),sha256=parent_hash,size_bytes=parent_bytes,width=w,height=h,frames=count),
            box=box, analysis=dict(video=str(final),sha256=paths.sha(partial),size_bytes=partial.stat().st_size,width=size[0],height=size[1],frames=count,codec='FFV1'),
            timing=dict(kind='validated_cfr',fps=fps,start_pts_seconds=start_pts/fps,frame_indices='unchanged'),
            raw_analysis_to_parent=_mapping(box,size))
        bounded()
        os.link(partial,final);partial.unlink()  # Atomic publication, refusing an existing destination.
        return final,metadata
    finally:
        cap.release()
        if writer is not None:writer.release()
        if partial.exists():partial.unlink()


def validate(source, region):
    """Reuse an already analyzed source, never crop it again; fail on stale identities/mapping."""
    import cv2
    try:
        if region['schema'] != 'image-lens.source-region/1':raise ValueError('unknown source-region schema')
        parent,analysis = region['parent'],region['analysis'];box = parse_box(region['box'])
        source = paths.confined(source,exists=True,max_bytes=MAX_BYTES)
        original = paths.confined(parent['video'],exists=True,max_bytes=MAX_BYTES)
        if (str(source) != analysis['video'] or paths.sha(source) != analysis['sha256']
                or source.stat().st_size != analysis['size_bytes'] or analysis['codec'] != 'FFV1'
                or paths.sha(original) != parent['sha256'] or original.stat().st_size != parent['size_bytes']):
            raise ValueError('source-region parent/analysis identity mismatch')
        if box[2] > parent['width'] or box[3] > parent['height']:raise ValueError('source-region parent geometry mismatch')
        for path,info in ((source,analysis),(original,parent)):
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened() or [int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_COUNT)] != [info['width'],info['height'],info['frames']]:
                    raise ValueError('source-region stored geometry mismatch')
                if abs(cap.get(cv2.CAP_PROP_FPS)-region['timing']['fps'])>1e-6:raise ValueError('source-region timing mismatch')
            finally:cap.release()
        if (analysis['frames'] != parent['frames'] or region['timing']['kind'] != 'validated_cfr'
                or region['raw_analysis_to_parent'] != _mapping(box,(analysis['width'],analysis['height']))):
            raise ValueError('source-region mapping/timing mismatch')
    except (KeyError,TypeError,ZeroDivisionError) as error:
        raise ValueError('invalid source-region provenance') from error
    return copy.deepcopy(region)
