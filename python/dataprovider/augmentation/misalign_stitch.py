import augmentor
import random
import numpy as np
from scipy.sparse import dok_matrix
from scipy.ndimage.filters import gaussian_filter


class Occlude(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.5):
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
        location=random.randint(20, )
        raw, label = self.misalign_stitch(raw, label)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample

    def misalign_stitch(self, raw, label):
        

def shear3d(volume, shear=20, axis=1):
    result = volume.copy()
    shift_list = np.around(np.linspace(-shear//2, shear//2, num=result.shape[2])).astype(np.int8)
    for index, shift in enumerate(shift_list):
        result[:, :, index] = np.roll(result[:, :, index], shift, axis=axis)

    return result

def discontinuous_stitch(raw, label, error=(0, 20, 0), location=100):
    # Initialize distorted raw volume and label
    err_z, err_y, err_x = error
    z_len, y_len, x_len = raw.shape

    distorted_raw = np.zeros((z_len-err_z,
                              y_len-err_y,
                              x_len-err_x)).astype(raw.dtype)

    distorted_label = np.zeros((z_len-err_z,
                                y_len-err_y,
                                x_len-err_x)).astype(label.dtype)

    # Shear raw volume
    distorted_raw[:, :, location:] = raw[:, :-err_y, location:]
    distorted_raw[:, :, :location] = raw[:, err_y:, :location]

    distorted_label[:, :, location:] = label[:, :-err_y, location:]
    distorted_label[:, :, :location] = label[:, err_y:, :location]

    # Shear label
    fill_region = label[:, err_y//2:-err_y//2,
                        location-10:location+10]
    fill_region = shear3d(fill_region, shear=err_y, axis=1)
    distorted_label[:, :, location-10:location+10] = fill_region

    return distorted_raw, distorted_label

