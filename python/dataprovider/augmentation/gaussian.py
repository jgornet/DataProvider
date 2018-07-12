import numpy as np
from scipy.ndimage import gaussian_filter
import augmentor
from random import random


class Gaussian(augmentor.DataAugment):
    def __init__(self, sigma_max=(0.1666, 1.0), skip_ratio=0.3):
        self.setSigmaMax(sigma_max)
        self.setSkipRatio(skip_ratio)

    def __call__(self, sample, **kwargs):
        if random() > self.skip_ratio:
            sample = self.augment(sample, **kwargs)
        return sample

    def augment(self, sample, **kwargs):
        sigma = tuple([random()*i_sigma_max for i_sigma_max in sigma_max])
        sample = gaussian_filter(sample, sigma=sigma)

        return sample

    def setSigmaMax(self, sigma_max):
        if not isinstance(sigma_max, tuple) or len(sigma_max) != 2:
            raise ValueError("sigma_max must be a tuple of length two")
        if sigma_max[0] <= 0 or sigma_max[1] <= 0:
            raise ValueError("sigma_max must be greater than zero")
        self.sigma_max = sigma_max

    def setSkipRatio(self, skip_ratio):
        if skip_ratio < 0 or skip_ratio > 1:
            raise ValueError("skip_ratio must be a number between 0 and 1")
        return skip_ratio
