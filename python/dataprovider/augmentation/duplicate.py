import augmentor
import random
import numpy as np


class Duplicate(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.8):
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
        slices = random.randrange(1, 5)
        location = random.randrange(0, raw.shape[0]-slices)
        raw, label = duplication_augmentation(raw, label, location=location,
                                              slices=slices)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample


def duplication_augmentation(raw, label, location=20, slices=3):
    distorted_raw = raw.copy()
    duplicate_slices = np.repeat(raw[location, :, :].reshape(1, raw.shape[1], raw.shape[2]), slices, axis=0)
    distorted_raw[location:location+slices, :, :] = duplicate_slices

    return distorted_raw, label
