# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from mmengine.model import BaseModel
from mmseg.registry import MODELS


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    "pretrain_mae_base_patch16_224",
    "pretrain_mae_large_patch16_224",
    "pretrain_mae_base_patch16",
    "pretrain_mae_large_patch16",
    "pretrain_mae_small_patch16",
]


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_learnable_pos_emb=False,
        add_cls_token=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.add_cls_token = add_cls_token
        # TODO: Add the cls token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_embed = nn.Embedding(1, embed_dim)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)
        if add_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
            trunc_normal_(self.cls_embed.weight, std=0.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "cls_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask):
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(
                x.shape[0], 1, -1
            ) + self.cls_embed.weight.type_as(x).to(x.device).clone().detach().reshape(
                1, 1, -1
            )
            x = torch.cat((cls_tokens, x), dim=1)
            # x = self.cls_embed(x[:, 0])
        B, _, C = x.shape
        if self.add_cls_token:
            mask = torch.cat([mask, torch.zeros(B, 1).type_as(mask)], dim=1)
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
    ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * patch_size**2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

        return x


class PretrainVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        fusion_depth=0,
        add_cls_token=False,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_ema_teacher=False,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        **kwargs,
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            add_cls_token=add_cls_token,
        )
        if use_ema_teacher:
            self.ema_teacher = PretrainVisionTransformerEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=encoder_in_chans,
                num_classes=encoder_num_classes,
                embed_dim=encoder_embed_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                use_learnable_pos_emb=use_learnable_pos_emb,
                add_cls_token=add_cls_token,
            )
        else:
            self.ema_teacher = None
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )
        if fusion_depth > 0:
            self.fusion_decoder = PretrainVisionTransformerDecoder(
                patch_size=patch_size,
                num_patches=self.encoder.patch_embed.num_patches,
                num_classes=0,
                embed_dim=decoder_embed_dim,
                depth=fusion_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
            )
        else:
            self.fusion_decoder = None

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )
        if decoder_depth > 0:
            self.fusion_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.mask1_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask2_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.classifis_token = nn.Embedding(1000, decoder_embed_dim)

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        if add_cls_token:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(encoder_embed_dim),
                nn.Linear(encoder_embed_dim, decoder_embed_dim),
                nn.GELU(),
                nn.Linear(decoder_embed_dim, 2),
            )

        trunc_normal_(self.mask1_token, std=0.02)
        trunc_normal_(self.mask2_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def frozzen_ema_teacher(self):
        if self.ema_teacher is not None:
            self.ema_teacher.eval()
            for param in self.ema_teacher.parameters():
                param.requires_grad = False

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x1, x2, mask1, mask2):
        # x1, x2 = torch.split(x1, [3, 3], dim=1)
        x1_vis = self.encoder(x1, mask1)  # [B, N_vis, C_e]
        x2_vis = self.encoder(x2, mask2)  # [B, N_vis, C_e]

        if self.encoder.add_cls_token:
            x1_vis_t1 = x1_vis[:, 1:]
            x2_vis_t1 = x2_vis[:, 1:]
            x1_cls = x1_vis[:, 0]
            x2_cls = x2_vis[:, 0]
            x1x2_cls = torch.cat([x1_cls, x2_cls], dim=0)
            x1_vis = x1_cls.unsqueeze(1).detach() + x1_vis_t1
            x2_vis = x2_cls.unsqueeze(1).detach() + x2_vis_t1
            x1x2_cls_pred = self.cls_head(x1x2_cls)
            gt_cls = torch.cat(
                [
                    torch.zeros(x1_cls.shape[0]).type_as(x1_cls),
                    torch.ones(x2_cls.shape[0]).type_as(x2_cls),
                ],
                dim=0,
            )
            gt_cls_label = gt_cls.long()
        else:
            x1x2_cls_pred = None
            gt_cls_label = None
        x1_vis = self.encoder_to_decoder(x1_vis)  # [B, N_vis, C_d]
        x2_vis = self.encoder_to_decoder(x2_vis)  # [B, N_vis, C_d]
        B, _, C = x1_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.

        if self.fusion_decoder is not None:
            pos_embed = (
                self.pos_embed.type_as(x1).to(x1.device).clone().detach().squeeze(0)
            )
            target_x1 = []
            target_x2 = []
            for i in range(B):
                m1 = mask1[i]
                m2 = mask2[i]
                pos_emb_vis1 = pos_embed[~m1].reshape(-1, C)
                pos_emb_vis2 = pos_embed[~m2].reshape(-1, C)
                pos_emb_visf = pos_embed[(~m1) | (~m2)].reshape(-1, C)
                # cls_emb_vis1 = self.classifis_token.weight[0, :].reshape(1, C)
                # cls_emb_vis2 = self.classifis_token.weight[1, :].reshape(1, C)
                # cls_emb_visf = self.classifis_token.weight[2, :].reshape(1, C)
                x_fusion = torch.cat(
                    [
                        x1_vis[i] + pos_emb_vis1,
                        x2_vis[i] + pos_emb_vis2,
                        self.fusion_token + pos_emb_visf,
                    ],
                    dim=0,
                )

                x_fusion = self.fusion_decoder(
                    x_fusion.unsqueeze(0), pos_emb_visf.shape[0]
                )
                pos_emb_visf = pos_emb_visf.unsqueeze(0)
                pos_embed_bnc = pos_embed.reshape(1, -1, C)
                x1_full = torch.cat(
                    [
                        x_fusion + pos_emb_visf,
                        self.mask1_token + pos_embed_bnc,
                    ],
                    dim=1,
                )
                x2_full = torch.cat(
                    [x_fusion + pos_emb_visf, self.mask2_token + pos_embed_bnc], dim=1
                )
                # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
                tx1 = self.decoder(
                    x1_full, pos_embed_bnc.shape[1]
                )  # [B, N_mask, 3 * 16 * 16]
                tx2 = self.decoder(
                    x2_full, pos_embed_bnc.shape[1]
                )  # [B, N_mask, 3 * 16 * 16]
                target_x1.append(tx1)
                target_x2.append(tx2)
            x1 = torch.cat(target_x1, dim=0)
            x2 = torch.cat(target_x2, dim=0)
        else:
            expand_pos_embed = (
                self.pos_embed.expand(B, -1, -1)
                .type_as(x1)
                .to(x1.device)
                .clone()
                .detach()
            )
            pos_emb_vis1 = expand_pos_embed[~mask1].reshape(B, -1, C)
            pos_emb_vis2 = expand_pos_embed[~mask2].reshape(B, -1, C)
            pos_emb_bnc = expand_pos_embed.reshape(B, -1, C)
            x1_full = torch.cat(
                [
                    x1_vis + pos_emb_vis1,
                    x2_vis + pos_emb_vis2,
                    self.mask1_token + pos_emb_bnc,
                ],
                dim=1,
            )
            x1 = self.decoder(x1_full, pos_emb_bnc.shape[1])
            x2_full = torch.cat(
                [
                    x1_vis + pos_emb_vis1,
                    x2_vis + pos_emb_vis2,
                    self.mask2_token + pos_emb_bnc,
                ],
                dim=1,
            )
            x2 = self.decoder(x2_full, pos_emb_bnc.shape[1])

        return (x1, x2, x1x2_cls_pred, gt_cls_label)

    def update_ema(self, decay=0.9999):
        if self.ema_teacher is not None:
            with torch.no_grad():
                for p1, p2 in zip(
                    self.encoder.parameters(), self.ema_teacher.parameters()
                ):
                    p2.copy_(p1.detach() * decay + p2 * (1 - decay))


@MODELS.register_module()
class PretrainViT(BaseModel):
    def __init__(self, init_cfg=None, data_preprocessor=None, **kwargs):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.model = PretrainVisionTransformer(**kwargs)

    def forward(self, x):
        x1, x2, mask1, mask2 = torch.split(x, [3, 3, 1, 1], dim=1)


@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_small_patch16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=8,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_base_patch16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_large_patch16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=24,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
