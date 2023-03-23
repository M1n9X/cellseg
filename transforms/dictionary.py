# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Any, Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import torch
from monai.config import KeysCollection, SequenceStr
from monai.config.type_definitions import (DtypeLike, KeysCollection,
                                           NdarrayOrTensor)
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (CenterSpatialCrop, InvertibleTransform, Padd,
                              RandSpatialCrop, SpatialPad)
from monai.transforms.transform import MapTransform
from monai.utils import Method, PytorchPadMode, TraceKeys

from .array import (GenerateDistanceMap, GenerateInstanceBorder,
                    GenerateWatershedMarkers, GenerateWatershedMask, Watershed)

__all__ = [
    "WatershedD",
    "WatershedDict",
    "Watershedd",
    "GenerateWatershedMaskD",
    "GenerateWatershedMaskDict",
    "GenerateWatershedMaskd",
    "GenerateInstanceBorderD",
    "GenerateInstanceBorderDict",
    "GenerateInstanceBorderd",
    "GenerateDistanceMapD",
    "GenerateDistanceMapDict",
    "GenerateDistanceMapd",
    "GenerateWatershedMarkersD",
    "GenerateWatershedMarkersDict",
    "GenerateWatershedMarkersd",
    "ResizeWithPadOrRandCropd",
    "ResizeWithPadOrRandCrop",
]


class Watershedd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.Watershed`.
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        mask_key: keys of mask used in watershed. Only points at which mask == True will be labeled.
        markers_key: keys of markers used in watershed. If None (no markers given), the local minima of the image are
            used as markers.
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate neighbors
            for connection. Following the scipy convention, default is a one-connected array of the dimension of the
            image.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `image` shape is not [1, H, W].
        ValueError: when the `mask` shape is not [1, H, W].

    """

    backend = Watershed.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: Optional[str] = "mask",
        markers_key: Optional[str] = None,
        connectivity: Optional[int] = 1,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.markers_key = markers_key
        self.transform = Watershed(connectivity=connectivity, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        markers = d[self.markers_key] if self.markers_key else None
        mask = d[self.mask_key] if self.mask_key else None

        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], mask, markers)

        return d


class GenerateWatershedMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
        mask_key: the mask will be written to the value of `{mask_key}`.
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateWatershedMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.transform = GenerateWatershedMask(
            softmax=softmax,
            sigmoid=sigmoid,
            threshold=threshold,
            remove_small_objects=remove_small_objects,
            min_size=min_size,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.transform(d[key])
            key_to_add = f"{self.mask_key}"
            if key_to_add in d:
                raise KeyError(f"Mask with key {key_to_add} already exists.")
            d[key_to_add] = mask
        return d


class GenerateInstanceBorderd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateInstanceBorder`.

    Args:
        keys: keys of the corresponding items to be transformed.
        hover_map_key: keys of hover map used to generate probability map.
        border_key: the instance border map will be written to the value of `{border_key}`.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `hover_map` has only one value.
        ValueError: when the `sobel gradient map` has only one value.

    """

    backend = GenerateInstanceBorder.backend

    def __init__(
        self,
        keys: KeysCollection,
        hover_map_key: str = "hover_map",
        border_key: str = "border",
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_map_key = hover_map_key
        self.border_key = border_key
        self.transform = GenerateInstanceBorder(
            kernel_size=kernel_size, remove_small_objects=remove_small_objects, min_size=min_size, dtype=dtype
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            instance_border = self.transform(d[key], d[self.hover_map_key])
            key_to_add = f"{self.border_key}"
            if key_to_add in d:
                raise KeyError(
                    f"Instance border map with key {key_to_add} already exists.")
            d[key_to_add] = instance_border
        return d


class GenerateDistanceMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateDistanceMap`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate distance map.
        dist_key: the distance map will be written to the value of `{dist_key}`.
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify
            callable functions for smoothing.
            For example, if you want apply gaussian smooth, you can specify `smooth_fn = GaussianSmooth()`
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateDistanceMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        dist_key: str = "dist",
        smooth_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.dist_key = dist_key
        self.transform = GenerateDistanceMap(smooth_fn=smooth_fn, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            distance_map = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.dist_key}"
            if key_to_add in d:
                raise KeyError(
                    f"Distance map with key {key_to_add} already exists.")
            d[key_to_add] = distance_map
        return d


class GenerateWatershedMarkersd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMarkers`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate markers.
        markers_key: the markers will be written to the value of `{markers_key}`.
        threshold: threshold the float values of instance border map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateWatershedMarkers.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        markers_key: str = "markers",
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.markers_key = markers_key
        self.transform = GenerateWatershedMarkers(
            threshold=threshold,
            radius=radius,
            min_size=min_size,
            remove_small_objects=remove_small_objects,
            postprocess_fn=postprocess_fn,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            markers = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.markers_key}"
            if key_to_add in d:
                raise KeyError(
                    f"Markers with key {key_to_add} already exists.")
            d[key_to_add] = markers
        return d


class ResizeWithPadOrRandCrop(InvertibleTransform):
    """
    Resize an image to a target spatial size by either centrally cropping the image or
    padding it evenly with a user-specified mode.
    When the dimension is smaller than the target size, do symmetric padding along that dim.
    When the dimension is larger than the target size, do central cropping along that dim.
    Args:
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = list(set(SpatialPad.backend) & set(RandSpatialCrop.backend))

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        self.padder = SpatialPad(
            spatial_size=spatial_size, method=method, mode=mode, **pad_kwargs)
        self.cropper = RandSpatialCrop(
            roi_size=spatial_size, random_size=False)

    # type: ignore
    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, **pad_kwargs) -> torch.Tensor:
        """
        Args:
            img: data to pad or crop, assuming `img` is channel-first and
                padding or cropping doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.
        """
        orig_size = img.shape[1:]
        ret = self.padder(self.cropper(img), mode=mode, **pad_kwargs)
        # remove the individual info and combine
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            pad_info = ret_.applied_operations.pop(-1)
            crop_info = ret_.applied_operations.pop(-1)
            self.push_transform(ret_, orig_size=orig_size, extra_info={
                                "pad_info": pad_info, "crop_info": crop_info})
        return ret

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(img)
        return self.inverse_transform(img, transform)

    def inverse_transform(self, img: MetaTensor, transform) -> MetaTensor:
        # we joined the cropping and padding, so put them back before calling the inverse
        crop_info = transform[TraceKeys.EXTRA_INFO].pop("crop_info")
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(crop_info)
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return self.cropper.inverse(inv)


class ResizeWithPadOrRandCropd(Padd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ResizeWithPadOrRandCrop`.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        method: str = Method.SYMMETRIC,
        **pad_kwargs,
    ) -> None:
        padcropper = ResizeWithPadOrRandCrop(
            spatial_size=spatial_size, method=method, **pad_kwargs)
        super().__init__(keys, padder=padcropper, mode=mode,
                         allow_missing_keys=allow_missing_keys)  # type: ignore


WatershedD = WatershedDict = Watershedd
GenerateWatershedMaskD = GenerateWatershedMaskDict = GenerateWatershedMaskd
GenerateInstanceBorderD = GenerateInstanceBorderDict = GenerateInstanceBorderd
GenerateDistanceMapD = GenerateDistanceMapDict = GenerateDistanceMapd
GenerateWatershedMarkersD = GenerateWatershedMarkersDict = GenerateWatershedMarkersd
