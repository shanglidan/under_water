import mmcv

from ..registry import PIPELINES
from .compose import Compose
import pdb


@PIPELINES.register_module
class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale, flip=False, vflip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple)
        self.flip = flip
        self.vflip = vflip

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        vflip_aug = [True] if self.vflip else [False]
        # print(results.keys())
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = flip
                _results['vflip'] = False
                _results['test'] = True
                # print(_results.keys())
                data = self.transforms(_results)
                aug_data.append(data)
            if self.vflip == True:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = False
                _results['vflip'] = self.vflip
                _results['test'] = True
                
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={}, vflip={})'.format(
            self.transforms, self.img_scale, self.flip, self.vflip)
        return repr_str
