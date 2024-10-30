# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import tinygrad
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa


inf = float('inf')


def load_model(model_path):
    ckpt = tinygrad.nn.state.torch_load(model_path)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    args = args[:-1] + ', landscape_only=False)'
    from pprint import pprint
    pprint(args)
    net = eval(args)
    new_ckpt = dict(ckpt['model'])
    if not any(k.startswith('dec_blocks2') for k in ckpt):
        for key, value in ckpt.items():
            if key.startswith('dec_blocks'):
                new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
    tinygrad.nn.state.load_state_dict(net, new_ckpt, strict=False)
    return net


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)