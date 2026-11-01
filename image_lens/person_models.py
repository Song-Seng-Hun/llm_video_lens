"""Explicit pinned local person detectors. Heavy dependencies stay lazy; no downloads."""
from pathlib import Path
from image_lens import paths

PINS = {
    'humanart': (101400344, '3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23'),
    'deepghs': (44583231, '6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90'),
    'sam3': (3402867661, '0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173'),
    'tokenizer': (1356917, '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'),
}
DEEPGHS_RECIPE = dict(input=[640, 640], color='RGB', resize='PIL bicubic stretch', scale='1/255',
                      confidence_strictly_above=.3, nms='inclusive pixel area', nms_threshold=.5,
                      coordinates='native resize, round, clip', provider='CPUExecutionProvider', cpu_threads=4,
                      maximum_instances=32, overflow='stable highest-score32; native-frame diagnostic')
SAM3_RECIPE = dict(prompt='person', input=[1008, 1008], confidence_strictly_above=.5,
                   boxes='native xyxy clipped to frame', coverage='DeepGHS zero positive-area intersection with every original SAM3 box',
                   masks='no pixel filtering', strict_checkpoint=True, inactive_neck_keys=22, generated_buffers=5)


def _asset(path, key):
    path = Path(path).expanduser().absolute()
    paths._no_symlinks(path)
    size, digest = PINS[key]
    if not path.is_file() or path.stat().st_size != size or paths.sha(path) != digest:
        raise ValueError(key + ' model pin mismatch; supply the pinned local asset')
    return dict(path=str(path), sha256=digest, bytes=size)


def resolve(path):
    """Validate every selected asset before any model/runtime import or construction."""
    path = Path(path).expanduser().absolute()
    paths._no_symlinks(path)
    if path.is_dir() or (path.is_file() and path.stat().st_size == PINS['sam3'][0]):
        weight = path/'model.safetensors' if path.is_dir() else path
        primary = _asset(weight, 'sam3')
        tokenizer = _asset(weight.parent/'bpe_simple_vocab_16e6.txt.gz', 'tokenizer')
        supplement = _asset(weight.parent/'anime-person.onnx', 'deepghs')
        return dict(kind='sam3-deepghs', **primary, primary='sam3', supplement=supplement, tokenizer=tokenizer,
                    source_commit='9eacae749049e2b864fe63f90f3dd17c689bee03',
                    model_revision='b72a14d8127e17e6f2a3d2e075bbbf4307ba146e',
                    supplement_revision='e39c744c22432ad01f91dd254fe2b02c8d878b8c',
                    recipe=dict(SAM3_RECIPE, supplement=DEEPGHS_RECIPE))
    if not path.is_file():
        raise ValueError('pinned HumanArt/DeepGHS ONNX or SAM3 local bundle required')
    size = path.stat().st_size
    for key in ('humanart', 'deepghs'):
        if size == PINS[key][0]:
            info = dict(kind=key, **_asset(path, key))
            if key == 'deepghs':
                info.update(model_revision='e39c744c22432ad01f91dd254fe2b02c8d878b8c', recipe=DEEPGHS_RECIPE)
            return info
    raise ValueError('detector model pin mismatch; unknown local ONNX')


def add_disjoint(primary, primary_scores, secondary, secondary_scores):
    """Keep original primary boxes/scores; compare every proposal only with original primary."""
    import numpy as np
    extra = []
    for i, a in enumerate(secondary):
        if all(min(a[2], b[2]) <= max(a[0], b[0]) or
               min(a[3], b[3]) <= max(a[1], b[1]) for b in primary):
            extra.append(i)
    return (np.concatenate((primary, secondary[extra]), axis=0),
            np.concatenate((primary_scores, secondary_scores[extra]), axis=0))


def inclusive_nms(boxes, scores, threshold=.5):
    """DeepGHS source-compatible inclusive pixel areas and stable score ties."""
    import numpy as np
    area = np.maximum(boxes[:, 2]-boxes[:, 0]+1, 0)*np.maximum(boxes[:, 3]-boxes[:, 1]+1, 0)
    order = np.argsort(-scores, kind='stable'); kept = []
    while order.size:
        i = int(order[0]); kept.append(i); rest = order[1:]
        size = np.maximum(np.minimum(boxes[i, 2:], boxes[rest, 2:])-np.maximum(boxes[i, :2], boxes[rest, :2])+1, 0)
        intersection = size[:, 0]*size[:, 1]; union = area[i]+area[rest]-intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[overlap <= threshold]
    return np.asarray(kept, dtype=int)


def anime_boxes(output, width, height):
    """Decode fixed one-class YOLOv8 [1,5,N] at .3 confidence / .5 inclusive NMS."""
    import numpy as np
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[:2] != (1, 5) or not np.isfinite(output).all():
        raise ValueError('unexpected DeepGHS detector output')
    rows = output[0].T; rows = rows[rows[:, 4] > .3]
    boxes = np.concatenate((rows[:, :2]-rows[:, 2:4]/2, rows[:, :2]+rows[:, 2:4]/2), axis=1)
    good = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[good], rows[good, 4]
    keep = inclusive_nms(boxes, scores); boxes, scores = boxes[keep], scores[keep]
    boxes *= np.array([width/640, height/640, width/640, height/640])
    boxes = np.rint(boxes)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    good = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[good], scores[good]
    return boxes.astype(np.float32), scores.astype(np.float32)


class DeepGHS:
    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if (self.session.get_providers() != ['CPUExecutionProvider'] or len(inputs) != 1
                or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4
                or any(actual != expected and not (axis in (0, 2, 3) and isinstance(actual, str))
                       for axis, (actual, expected) in enumerate(zip(inputs[0].shape, (1, 3, 640, 640))))
                or len(self.session.get_outputs()) != 1):
            raise ValueError('unexpected DeepGHS provider/input/output contract')
        self.input_name = inputs[0].name

    def __call__(self, image):
        import numpy as np
        from PIL import Image
        rgb = Image.fromarray(image[:, :, ::-1])
        resized = np.asarray(rgb.resize((640, 640), Image.Resampling.BICUBIC), dtype=np.float32)/255.
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        output, = self.session.run(None, {self.input_name: tensor})
        boxes, scores = anime_boxes(output, image.shape[1], image.shape[0])
        self.diagnostics = []
        if len(boxes) > 32:
            count = len(boxes)
            keep = np.argsort(-scores, kind='stable')[:32]
            boxes, scores = boxes[keep], scores[keep]
            self.diagnostics = [dict(detector='deepghs', role='primary', original_count=count, kept_count=32,
                                     dropped_count=count-32, basis='post-NMS native valid boxes; stable highest-score32 resource bound')]
        return boxes, scores


class Sam3Coverage:
    def __init__(self, selection):
        import mlx.core as mx
        from image_lens._sam3_mlx.model_builder import build_sam3_image_model
        from image_lens._sam3_mlx.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(checkpoint_path=selection['path'], bpe_path=selection['tokenizer']['path'])
        self.processor = Sam3Processor(model, resolution=1008, confidence_threshold=.5)
        self.supplement = DeepGHS(selection['supplement']['path'])
        self.runtime = dict(mlx=mx.__version__, mlx_device=str(mx.default_device()))

    def __call__(self, image):
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        height, width = image.shape[:2]
        state = self.processor.set_image(Image.fromarray(image[:, :, ::-1]))
        state = self.processor.set_text_prompt('person', state)
        mx.eval(state['boxes'], state['scores'])
        boxes = np.array(state['boxes'], dtype=np.float32, copy=True); scores = np.asarray(state['scores'], dtype=np.float32)
        if boxes.shape != (len(scores), 4) or scores.ndim != 1 or len(scores) > 200 or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('invalid SAM3 detector output')
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        secondary, secondary_scores = self.supplement(image)
        self.diagnostics = [dict(item, role='supplement') for item in self.supplement.diagnostics]
        return add_disjoint(boxes, scores, secondary, secondary_scores)


def create(selection):
    if selection['kind'] == 'sam3-deepghs':
        return Sam3Coverage(selection)
    if selection['kind'] == 'deepghs':
        return DeepGHS(selection['path'])
    raise ValueError('HumanArt uses the existing pose backend')
