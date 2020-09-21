import torch
from pytracking.features.preprocessing import sample_patch
from pytracking import TensorList

import cv2


class ExtractorBase:
    """Base feature extractor class.
    args:
        features: List of features.
    """
    def __init__(self, features):
        self.features = features

    def initialize(self, extractor = None):
        for f in self.features:
            if extractor is not None:
                f.initialize(extractor)
            else:
                f.initialize()

    def free_memory(self):
        for f in self.features:
            f.free_memory()


class SingleResolutionExtractor(ExtractorBase):
    """Single resolution feature extractor.
    args:
        features: List of features.
    """
    def __init__(self, features):
        super().__init__(features)

        self.feature_stride = self.features[0].stride()
        if isinstance(self.feature_stride, (list, TensorList)):
            self.feature_stride = self.feature_stride[0]

    def stride(self):
        return self.feature_stride

    def size(self, input_sz):
        return input_sz // self.stride()

    def extract(self, im, pos, scales, image_sz):
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        im_patches = torch.cat([sample_patch(im, pos, s*image_sz, image_sz) for s in scales])

        # Compute features
        feature_map = torch.cat(TensorList([f.get_feature(im_patches) for f in self.features]).unroll(), dim=1)

        return feature_map


class MultiResolutionExtractor(ExtractorBase):
    """Multi-resolution feature extractor.
    args:
        features: List of features.
    """
    def __init__(self, features):
        super().__init__(features)
        self.is_color = None

    def stride(self):
        return TensorList([f.stride() for f in self.features if self._return_feature(f)]).unroll()

    def size(self, input_sz):
        return TensorList([f.size(input_sz) for f in self.features if self._return_feature(f)]).unroll()

    def dim(self):
        return TensorList([f.dim() for f in self.features if self._return_feature(f)]).unroll()

    def get_fparams(self, name: str = None):
        if name is None:
            return [f.fparams for f in self.features if self._return_feature(f)]
        return TensorList([getattr(f.fparams, name) for f in self.features if self._return_feature(f)]).unroll()

    def get_attribute(self, name: str, ignore_missing: bool = False):
        if ignore_missing:
            return TensorList([getattr(f, name) for f in self.features if self._return_feature(f) and hasattr(f, name)])
        else:
            return TensorList([getattr(f, name, None) for f in self.features if self._return_feature(f)])

    def get_unique_attribute(self, name: str):
        feat = None
        for f in self.features:
            if self._return_feature(f) and hasattr(f, name):
                if feat is not None:
                    raise RuntimeError('The attribute was not unique.')
                feat = f
        if feat is None:
            raise RuntimeError('The attribute did not exist')
        return getattr(feat, name)

    def _return_feature(self, f):
        return self.is_color is None or self.is_color and f.use_for_color or not self.is_color and f.use_for_gray

    def set_is_color(self, is_color: bool):
        self.is_color = is_color

    def extract(self, im, pos, scales, image_sz):
        """Extract features.
        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract features from.
            image_sz: Size to resize the image samples to before extraction.
        """
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        im_patches = torch.cat([sample_patch(im, pos, s*image_sz, image_sz) for s in scales])

        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()

        return feature_map

    def extract_transformed(self, im, pos, scale, image_sz, transforms):
        """Extract features from a set of transformed image samples.
        args:
            im: Image.
            pos: Center position for extraction.
            scale: Image scale to extract features from.
            image_sz: Size to resize the image samples to before extraction.
            transforms: A set of image transforms to apply.
        """

        # Get image patche
        im_patch = sample_patch(im, pos, scale*image_sz, image_sz)

        # Apply transforms
        im_patches = torch.cat([T(im_patch) for T in transforms])

        # import cv2
        # import numpy as np
        # cv2.namedWindow('Patch', cv2.WINDOW_AUTOSIZE)
        # for p in im_patches:
        #     p_ = p.permute(1, 2, 0).cpu().numpy()
        #     cv2.imshow('Patch', p_.astype(np.uint8))
        #     cv2.waitKey(0)

        # Compute features
        feature_map = TensorList([f.get_feature(im_patches) for f in self.features]).unroll()
        # [27, 1024, 16, 16]
        return feature_map
