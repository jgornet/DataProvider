import augmentor
import random
import numpy as np


class Crop(augmentor.DataAugment):
    def __init__(self, size=(10, 128, 128)):
        self.size = size
    
    def __call__(self, sample, **kwargs):
        sample = self.augment(sample, **kwargs)
        return sample

    def prepare(self, spec, **kwargs):
        # No change in spec.
        return spec

    def augment(self, sample, **kwargs):
        raw = sample["input"]
        raw = raw.reshape(raw.shape[1],
                          raw.shape[2],
                          raw.shape[3])
        label = sample["soma_label"]
        label = label.reshape(label.shape[1],
                              label.shape[2],
                              label.shape[3])
        location = tuple([random.randrange(0, input_size - output_size)
                          if input_size - output_size > 0 else 0
                          for input_size, output_size in
                          zip(raw.shape, self.size)])
        raw, label = self.crop(raw, label, location=location)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample

    def crop(self, raw, label, location=(0, 0, 0)):
        output_z, output_y, output_x = self.size
        raw_z, raw_y, raw_x = raw.shape
        z, y, x = location

        raw = raw[z:(z + output_z),
                  y:(y + output_y),
                  x:(x + output_x)]

        label_z, label_y, label_x = label.shape
        label = label[z:(z + output_z),
                  y:(y + output_y),
                  x:(x + output_x)]

        return raw, label
