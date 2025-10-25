#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import yaml
from dotmap import DotMap


def get_config(config_name='settings'):

    with open('exp_settings/{}.yaml'.format(config_name), 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config


# import yaml
# from types import SimpleNamespace as NS

# def _dict_to_ns(d):
#     """Recursively convert nested dicts to SimpleNamespace objects."""
#     if isinstance(d, dict):
#         return NS(**{k: _dict_to_ns(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [_dict_to_ns(i) for i in d]
#     else:
#         return d

# def get_config(config_name='settings'):
#     """Load settings/<name>.yaml and return a dot-accessible namespace."""
#     path = f"settings/{config_name}.yaml"
#     with open(path, "r") as f:
#         cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
#     cfg = _dict_to_ns(cfg_dict)
#     return cfg
