import augmentor
import random
import numpy as np
from scipy.ndimage.interpolation import rotate


class Stitch(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.7):
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

        raw_z, raw_y, raw_x = raw.shape
        label_z, label_y, label_x = label.shape

        angle = random.randrange(0, 10)

        raw = rotate(raw, angle, axes=(0, 1))
        label = rotate(label, angle, axes=(0, 1), order=0)

        location = random.randrange(10, raw.shape[2]-10)
        error = random.randrange(1, 20)
        raw, label = discontinuous_stitch(raw, label, error=error,
                                          location=location)


        raw = rotate(raw, -angle, axes=(0, 1))
        label = rotate(label, -angle, axes=(0, 1), order=0)

        dist_z, dist_y, dist_x = raw.shape
        raw = raw[(dist_z-raw_z+6)//2:(dist_z+raw_z-6)//2,
                  (dist_y-raw_y+20)//2:(dist_y+raw_y-20)//2,
                  (dist_x-raw_x)//2:(dist_x+raw_x)//2]

        dist_z, dist_y, dist_x = label.shape
        label = label[(dist_z-label_z+6)//2:(dist_z+label_z-6)//2,
                      (dist_y-label_y+20)//2:(dist_y+label_y-20)//2,
                      (dist_x-label_x)//2:(dist_x+label_x)//2]

        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample


def discontinuous_stitch(raw, label, error=20, location=100):
    # Initialize distorted raw volume and label
    z_len, y_len, x_len = raw.shape

    distorted_raw = np.zeros((z_len,
                              y_len-error,
                              x_len)).astype(raw.dtype)

    distorted_label = np.zeros((z_len,
                                y_len-error,
                                x_len)).astype(label.dtype)

    # Shear raw volume
    distorted_raw[:, :, location:] = raw[:, :-error, location:]
    distorted_raw[:, :, :location] = raw[:, error:, :location]

    distorted_label[:, :, location:] = label[:, :-error, location:]
    distorted_label[:, :, :location] = label[:, error:, :location]

    # Shear label
    fill_region = label[:, error//2:-error//2,
                        location-10:location+10]
    fill_region = shear3d(fill_region, shear=error, axis=1)
    distorted_label[:, :, location-10:location+10] = fill_region

    return distorted_raw, distorted_label

def shear3d(volume, shear=20, axis=1):
    result = volume.copy()
    shift_list = np.around(np.linspace(-shear//2, shear//2, num=result.shape[2])).astype(np.int8)
    for index, shift in enumerate(shift_list):
        result[:, :, index] = np.roll(result[:, :, index], shift, axis=axis)

    return result
