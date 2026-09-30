
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from segment_anything import sam_model_registry
from segment_anything.modeling import ImageEncoderViT, PromptEncoder, MaskDecoder, Sam, TwoWayTransformer

IMG_SIZE = 256
PATCH_SIZE = 16
GLOBAL_ATTN_INDEXES = [2, 5, 8, 11]


def _interp_pos_embed(old, new_grid):
    old = old.permute(0, 3, 1, 2)
    new = F.interpolate(old, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return new.permute(0, 2, 3, 1)


def _interp_rel_pos(old, new_len):
    old = old.T.unsqueeze(0)
    new = F.interpolate(old, size=new_len, mode="linear", align_corners=False)
    return new.squeeze(0).T


def build_small_sam(checkpoint_path, img_size=IMG_SIZE):
    """Rebuilds SAM ViT-B for a smaller input resolution (256 instead of 1024).
    Only pos_embed and the 4 global blocks' rel_pos_h/w are interpolated;
    everything else (qkv/mlp/norm weights, patch_embed, neck, prompt_encoder,
    mask_decoder) is shape-independent of img_size and loads unchanged."""
    full_sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    full_state = full_sam.image_encoder.state_dict()

    prompt_embed_dim = 256
    grid = img_size // PATCH_SIZE

    small_encoder = ImageEncoderViT(
        depth=12, embed_dim=768, img_size=img_size, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12, patch_size=PATCH_SIZE, qkv_bias=True,
        use_rel_pos=True, global_attn_indexes=GLOBAL_ATTN_INDEXES,
        window_size=14, out_chans=prompt_embed_dim,
    )
    new_state = small_encoder.state_dict()
    copied, interpolated, skipped = [], [], []
    for k, v_new in new_state.items():
        if k in full_state:
            v_old = full_state[k]
            if v_old.shape == v_new.shape:
                new_state[k] = v_old.clone(); copied.append(k)
            elif k == "pos_embed":
                new_state[k] = _interp_pos_embed(v_old, grid); interpolated.append(k)
            elif "rel_pos_h" in k or "rel_pos_w" in k:
                new_state[k] = _interp_rel_pos(v_old, v_new.shape[0]); interpolated.append(k)
            else:
                skipped.append(k)
        else:
            skipped.append(k)
    small_encoder.load_state_dict(new_state)

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim, image_embedding_size=(grid, grid),
        input_image_size=(img_size, img_size), mask_in_chans=16,
    )
    prompt_encoder.load_state_dict(full_sam.prompt_encoder.state_dict())

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256,
    )
    mask_decoder.load_state_dict(full_sam.mask_decoder.state_dict())

    small_sam = Sam(
        image_encoder=small_encoder, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder,
        pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0],
    )
    del full_sam
    torch.cuda.empty_cache()
    return small_sam, {"copied": len(copied), "interpolated": len(interpolated),
                        "skipped": len(skipped), "skipped_keys": skipped}


class _LoRA_qkv(nn.Module):
    """SAMed-style: wraps SAM's combined qkv Linear, adding a rank-r
    low-rank update only to the Q and V slices of the output (K is untouched)."""
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q, self.linear_b_q = linear_a_q, linear_b_q
        self.linear_a_v, self.linear_b_v = linear_a_v, linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[..., :self.dim] = qkv[..., :self.dim] + new_q
        qkv[..., -self.dim:] = qkv[..., -self.dim:] + new_v
        return qkv


class LoRA_Sam(nn.Module):
    """SAMed-style parameter-efficient adaptation: rank-r LoRA on Q,V of every
    image-encoder block (base image-encoder weights frozen); prompt-encoder and
    mask-decoder are left fully trainable (NOT LoRA-only, per protocol).
    Image-only, prompt-free forward: no external point/box/mask prompt is ever used."""
    def __init__(self, sam_model, r=4):
        super().__init__()
        assert r > 0
        self.w_As, self.w_Bs = [], []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for blk in sam_model.image_encoder.blocks:
            w_qkv = blk.attn.qkv
            dim = w_qkv.in_features
            w_a_q, w_b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
            w_a_v, w_b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
            self.w_As += [w_a_q, w_a_v]
            self.w_Bs += [w_b_q, w_b_v]
            blk.attn.qkv = _LoRA_qkv(w_qkv, w_a_q, w_b_q, w_a_v, w_b_v)

        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

        self.sam = sam_model

    def forward(self, images_1ch):
        images_3ch = images_1ch.repeat(1, 3, 1, 1)
        x = self.sam.preprocess(images_3ch)
        image_embeddings = self.sam.image_encoder(x)
        sparse_emb, dense_emb = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        return F.interpolate(low_res_masks, size=images_1ch.shape[-2:], mode="bilinear", align_corners=False)
