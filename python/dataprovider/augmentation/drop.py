import augmentor
import random
import numpy as np
from scipy.ndimage import zoom


class Drop(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.3):
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
        location = random.randrange(10, raw.shape[1]-10)
        dropped_slices = random.randrange(3, 5)*2
        raw, label = dropping_augmentation(raw, label, location=location,
                                           dropped_slices=dropped_slices)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample


def dropping_augmentation(raw, label, dropped_slices=30, location=100):
    # Initialize distorted raw volume and label
    distorted_raw = np.zeros((raw.shape[0], raw.shape[1]-dropped_slices, raw.shape[2]))
    distorted_label = np.zeros((raw.shape[0], raw.shape[1]-dropped_slices, raw.shape[2]))

    # Populate distorted raw volume and label
    distorted_raw[:, :location-dropped_slices//2, :] = raw[:, :location-dropped_slices//2, :]
    distorted_raw[:, location-dropped_slices//2:, :] = raw[:, location+dropped_slices//2:, :]

    distorted_label[:, :location-dropped_slices//2, :] = label[:, :location-dropped_slices//2, :]
    distorted_label[:, location-dropped_slices//2:, :] = label[:, location+dropped_slices//2:, :]

    # Interpolate the distorted label
    mag = 6/(dropped_slices+6)
    print(location)
    print(dropped_slices)
    print(label.shape)
    fill_region = label[:, location-dropped_slices/2-3:location+dropped_slices/2+3, :]
    fill_region = zoom(fill_region, zoom=(1, mag, 1))
    fill_region = (fill_region > 0)*255

    # Fill in distorted label with interpolation
    distorted_label[:, location-dropped_slices//2-6:location-dropped_slices//2+6, :] = fill_region

    return distorted_raw.astype(raw.dtype), distorted_label.astype(label.dtype)
