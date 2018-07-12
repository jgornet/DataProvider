import augmentor
import random
import numpy as np
from scipy.sparse import dok_matrix
from scipy.ndimage.filters import gaussian_filter


class Occlude(augmentor.DataAugment):
    def __init__(self, skip_ratio=0.85):
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
        psf_size = random.uniform(3, 20)
        psf_size = (psf_size/6.157, psf_size, psf_size)
        raw, label = self.occlude_neuron3d(raw, label, size=psf_size)
        sample["input"] = raw.reshape(1,
                                      raw.shape[0],
                                      raw.shape[1],
                                      raw.shape[2])
        sample["soma_label"] = label.reshape(1,
                                      label.shape[0],
                                      label.shape[1],
                                      label.shape[2])

        return sample

    def occlude_neuron3d(self, raw, label, size=(4, 20, 20)):
        # Find a random neuron voxel
        neuron = self.dok_volume(label)
        neuron_voxel = random.choice(list(neuron))

        # Get occluding region
        min_x = max(0, neuron_voxel[2]-50)
        min_y = max(0, neuron_voxel[1]-50)
        min_z = max(0, neuron_voxel[0]-8)

        max_x = min(raw.shape[2], neuron_voxel[2]+50)
        max_y = min(raw.shape[1], neuron_voxel[1]+50)
        max_z = min(raw.shape[0], neuron_voxel[0]+8)

        x_len = max_x-min_x
        y_len = max_y-min_y
        z_len = max_z-min_z

        psf = np.zeros((z_len, y_len, x_len))
        psf[z_len//2, y_len//2, x_len//2] = 1
        psf = gaussian_filter(psf, size)
        psf = psf/psf[z_len//2, y_len//2, x_len//2]

        # Get background statistics
        sub_raw = raw[min_z:max_z, min_y:max_y, min_x:max_x]
        sub_label = label[min_z:max_z, min_y:max_y, min_x:max_x].astype(np.bool)
        average = np.mean(sub_raw[~sub_label])
        stdev = np.std(sub_raw[~sub_label])

        # Occlude region
        filtered_raw = raw.copy()
        filtered_raw[min_z:max_z, min_y:max_y, min_x:max_x] = raw[min_z:max_z, min_y:max_y, min_x:max_x]*(1-psf) + \
                                                psf*np.random.normal(loc=average,
                                                                    scale=stdev,
                                                                    size=(z_len,
                                                                          y_len,
                                                                          x_len)).astype(np.uint16)

        return filtered_raw, label

    def dok_volume(self, volume):
        dok = []
        for index in range(volume.shape[0]):
            dok += [(index, point[0], point[1]) for point in list(dok_matrix(volume[index, :, :]).keys())]
        return dok
