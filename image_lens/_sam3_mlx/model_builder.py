import mlx.core as mx
import mlx.nn as nn

from image_lens._sam3_mlx.model.sam3_image import Sam3Image
from image_lens._sam3_mlx.model.text_encoder_ve import VETextEncoder
from image_lens._sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from image_lens._sam3_mlx.model.vitdet import ViT
from image_lens._sam3_mlx.model.position_encoding import PositionEmbeddingSine
from image_lens._sam3_mlx.model.necks import Sam3DualViTDetNeck
from image_lens._sam3_mlx.model.vl_combiner import SAM3VLBackbone
from image_lens._sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from image_lens._sam3_mlx.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from image_lens._sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from image_lens._sam3_mlx.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer
)
from image_lens._sam3_mlx.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper
)


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )

def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )

def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )

def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = lambda: TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dims=256,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dims=256,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = lambda: TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dims=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder

def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

def _create_segmentation_head():
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dims=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head

def _create_geometry_encoder():
    geo_pos_enc = _create_position_encoding()
    geo_layer = lambda: TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dims=256,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dims=256,
        ),
    )

    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder

def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
):
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "dot_prod_scoring": dot_prod_scoring
    }

    model = Sam3Image(**common_params)
    return model

def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24
    )

def _create_vision_backbone(
    compile_mode=None,
    enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:

    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)

    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity
    )
    return vit_neck

def _create_sam3_transformer(has_presence_token: bool = True):
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

def _sanitize_conv_layout(model, weights):
    from mlx.utils import tree_flatten
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    for k in list(weights.keys()):
        exp = expected.get(k)
        v = weights[k]
        if exp is None or v.ndim != 4 or tuple(v.shape) == exp:
            continue
        perm = (1, 2, 3, 0) if "dconv" in k else (0, 2, 3, 1)
        cand = mx.transpose(v, perm)
        if tuple(cand.shape) == exp:
            weights[k] = cand
    return weights


def load_checkpoint(model, checkpoint_path):
    """Exact audited checkpoint contract; no missing learned tensors are accepted."""
    from mlx.utils import tree_flatten
    expected = dict(tree_flatten(model.parameters()))
    weights = _sanitize_conv_layout(model, mx.load(str(checkpoint_path)))
    generated = {'backbone.language_backbone.encoder.attn_mask'} | {
        'backbone.vision_backbone.position_encoding.cache.' + str((n, n))
        for n in (31, 63, 126, 252)}
    inactive = set()
    for i in range(4):
        layers = ['conv_1x1', 'conv_3x3'] + (
            ['dconv_2x2_0', 'dconv_2x2_1'] if i == 0 else ['dconv_2x2'] if i == 1 else [])
        for layer in layers:
            for suffix in ('weight', 'bias'):
                inactive.add(f'backbone.vision_backbone.sam2_convs.{i}.{layer}.{suffix}')
    missing = set(expected) - set(weights)
    extra = set(weights) - set(expected)
    mismatch = [k for k in set(expected) & set(weights)
                if tuple(expected[k].shape) != tuple(weights[k].shape)]
    if missing != generated or extra != inactive or mismatch:
        raise ValueError('Unexpected SAM3 checkpoint keys/shapes')
    if not bool(mx.all(mx.stack([mx.all(mx.isfinite(v)) for v in weights.values()])).item()):
        raise ValueError('Nonfinite SAM3 checkpoint tensor')
    for key in inactive:
        del weights[key]
    for key in generated:
        value = expected[key]
        if key.endswith('.attn_mask'):
            causal = mx.triu(mx.full(value.shape, -float('inf')), k=1)
            if not bool(mx.all(value == causal).item()):
                raise ValueError('SAM3 causal buffer differs')
        elif not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError('Nonfinite SAM3 position buffer')
        weights[key] = value
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())


def build_sam3_image_model(*, bpe_path, checkpoint_path):
    """Build only the pinned local image model. Download/conversion paths are absent."""
    if bpe_path is None or checkpoint_path is None:
        raise ValueError('Explicit local SAM3 checkpoint and vocabulary required')
    compile = False
    enable_inst_interactivity = False
    enable_segmentation = True

    vision_encoder = _create_vision_backbone(
        compile_mode=compile, enable_inst_interactivity=enable_inst_interactivity
    )

    text_encoder = _create_text_encoder(bpe_path)

    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    transformer = _create_sam3_transformer()

    dot_product_scoring = _create_dot_product_scoring()

    segmentation_head = (
        _create_segmentation_head()
        if enable_segmentation
        else None
    )

    input_geometry_encoder = _create_geometry_encoder()

    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring=dot_product_scoring
    )

    load_checkpoint(model, f"{checkpoint_path}")

    model.eval()

    return model
