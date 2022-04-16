from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import yaml
import os
import torch.nn as nn

class flatten(nn.Module):
    def __init__(self, backbone, input_type='BGR'):
        super(flatten, self).__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()

        self.input_type = input_type

    def forward(self, input):
        if self.input_type == 'BGR':
            input = input[:, [2,1,0], :, :]

        out = self.backbone(input)
        out = self.pool(out['res5'])
        out = self.flat(out)
        return out

class detectron2_manager(object):
    def __init__(self, model_path, base):
        self.model_path = model_path
        self.base = base

        # Configure File
        self.config_path = model_path.replace('.pkl', '.yaml')
        self.config = self.setup_cfg()

    def setup_cfg(self):
        # load config from file and command-line arguments
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['_BASE_'] = os.path.join(self.base, config_data['_BASE_'])
        config_data['MODEL']['WEIGHTS'] = os.path.join(self.base, config_data['MODEL']['WEIGHTS'])

        with open(self.config_path.replace('.yaml', '_new.yaml'), "w") as f:
            yaml.dump(config_data, f)

        self.config_path = self.config_path.replace('.yaml', '_new.yaml')

        cfg = get_cfg()
        cfg.merge_from_file(self.config_path)
        return cfg

    def load_detectron2_model(self):
        # Load Model
        model = build_model(self.config)

        # Load Weight
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.model_path)

        # Backbone
        backbone = model.backbone.bottom_up
        self.model = flatten(backbone, input_type=self.config.INPUT.FORMAT).to('cpu')