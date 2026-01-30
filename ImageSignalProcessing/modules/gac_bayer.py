# File: gac_bayer.py
# Description: Gamma Correction for Bayer (Raw) images
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)
# Modified: Tejus - Apply gamma directly on Bayer data

import numpy as np

from .basic_module import BasicModule


class GAC_BAYER(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gain = np.array(self.params.gain, dtype=np.uint32)  # x256
        x = np.arange(self.cfg.saturation_values.hdr + 1)
        lut = (
            (x / self.cfg.saturation_values.hdr) ** self.params.gamma
        ) * self.cfg.saturation_values.sdr
        self.lut = lut.astype(np.uint8)

    def execute(self, data):
        bayer_image = data["bayer"].astype(np.uint32)

        gac_bayer_image = np.right_shift(self.gain * bayer_image, 8)
        gac_bayer_image = np.clip(gac_bayer_image, 0, self.cfg.saturation_values.hdr)
        gac_bayer_image = self.lut[gac_bayer_image]

        data["bayer"] = gac_bayer_image
