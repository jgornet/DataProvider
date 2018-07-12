import augmentor
import random
import numpy as np


class Brightness(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.9):
        self.skip_ratio = skip_ratio

    def __call__(self, sample, **kwargs):
        if random.random() > self.skip_ratio:
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
        raw, label = brightness_augmentation(raw, label)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample


def brightness_augmentation(raw, label, maximum=0.05):
    distorted_raw = raw
    brightness = random.uniform(0, maximum)
    distorted_raw = distorted_raw + brightness

    return distorted_raw, label
