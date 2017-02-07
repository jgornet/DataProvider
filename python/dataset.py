#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from collections import OrderedDict
import copy
import numpy as np

from box import Box
from config_data import ConfigData, ConfigLabel
from vector import Vec3d

class Dataset(object):
    """
    Dataset interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError


class VolumeDataset(Dataset):
    """
    Dataset for volumetric data.

    Attributes:
        _data:  Dictionary mapping layer's name to TensorData, which contains
                4D volumetric data (e.g. EM image stacks, label stacks, etc.).

        _label: List of label layer's name.

        _spec:  Net specification. Dictionary mapping layer's name to its input
                dimension (either 3-tuple or 4-tuple).

        _range: Range of valid coordinates for accessing data given a net spec.
                It depends both on data and net specs.

        params: Dataset-specific parameters.
    """

    def __init__(self, config, **kwargs):
        """Initialize VolumeDataset."""
        self.build_from_config(config)
        # Set dataset-specific params.
        self.params = dict()
        for k, v in kwargs.iteritems():
            self.params[k] = v

    def __getattr__(self, name):
        """Access dataset-specific params."""
        assert name in self.params
        return self.params[name]

    def build_from_config(self, config):
        """
        Build dataset from a ConfiParser object generated by Parser's
        parse_dataset method.
        """
        self._reset()

        # First pass for images and labels.
        assert config.has_section('dataset')
        for name, data in config.items('dataset'):
            assert config.has_section(data)
            if '_mask' in data:
                # Mask will be processed later.
                continue
            if 'label' in data:
                self._data[name] = ConfigLabel(config, data)
                self._label.append(name)
            else:
                self._data[name] = ConfigData(config, data)
                self._image.append(name)

        # Second pass for masks.
        for name, data in config.items('dataset'):
            if '_mask' in data:
                if config.has_option(data, 'shape'):
                    # Lazy filling of mask shape. Since the shape of mask should
                    # be the same as the shape of corresponding label, it can be
                    # known only after having processed label in the first pass.
                    label, _ = data.split('_mask')
                    shape = self._data[label].shape()
                    config.set(data, 'shape', shape)
                self._data[name] = ConfigData(config, data)

        # Set dataset spec.
        spec = dict()
        for name, data in self._data.iteritems():
            spec[name] = tuple(data.fov())
        self.set_spec(spec)

    def get_spec(self):
        """Return dataset spec."""
        # TODO(kisuk):
        #   spec's value type is tuple, which is immutable. Do we still need to
        #   deepcopy it?
        return copy.deepcopy(self._spec)

    def get_imgs(self):
        return copy.deepcopy(self._image)

    def set_spec(self, spec):
        """Set spec and update valid range."""
        # Order by key
        self._spec = OrderedDict(sorted(spec.items(), key=lambda x: x[0]))
        self._update_range()

    def num_sample(self):
        """Return number of samples in valid range."""
        s = self._range.size()
        return s[0]*s[1]*s[2]

    def get_range(self):
        """Return valid range."""
        return Box(self._range)

    def get_sample(self, pos):
        """Extract a sample centered on pos.

        Every data in the sample is guaranteed to be center-aligned.

        Args:
            pos: Center coordinate of the sample.

        Returns:
            sample:     Dictionary mapping input layer's name to data.
            transform:  Dictionary mapping label layer's name to the type of
                        label transformation specified by user.

        """
        # self._spec is guaranteed to be ordered by key, so using OrderedDict
        # and iterating over already-sorted self._spec together guarantee the
        # sample is sorted.
        sample = OrderedDict()
        for name in self._spec.keys():
            sample[name] = self._data[name].get_patch(pos)

        transform = dict()
        for name in self._label:
            transform[name] = self._data[name].get_transform()

        return sample, transform

    def next_sample(self, spec=None):
        """Fetch next sample in a sample sequence."""
        return self.random_sample(spec)  # Currently just pick randomly.

    def random_sample(self, spec=None):
        """Fetch sample randomly"""
        # Dynamically change spec.
        if spec is not None:
            original_spec = self._spec
            self.set_spec(spec)

        try:
            pos = self._random_location()
            ret = self.get_sample(pos)
        except:
            # Return to original spec.
            if spec is not None:
                self.set_spec(original_spec)
            raise

        # Return to original spec.
        if spec is not None:
            self.set_spec(original_spec)

        # ret is a 2-tuple (sample, transform).
        return ret

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _reset(self):
        """Reset all attributes."""
        self._data  = dict()
        self._image = list()
        self._label = list()
        self._spec  = None
        self._range = None

    def _random_location(self):
        """Return one of the valid locations randomly."""
        s = self._range.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        # Global coordinate system.
        return Vec3d(z,y,x) + self._range.min()
        # DEBUG
        #return self._range.min()

    def _update_range(self):
        """
        Update valid range. It's computed by intersecting the valid range of
        each TensorData.
        """
        self._range = None
        for name, dim in self._spec.iteritems():
            # Update patch size.
            self._data[name].set_fov(dim[-3:])
            # Update valid range.
            r = self._data[name].range()
            if self._range is None:
                self._range = r
            else:
                self._range = self._range.intersect(r)
