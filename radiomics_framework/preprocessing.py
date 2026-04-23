from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from radiomics_framework.config import PreprocessingConfig


def resample_to_reference(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
    *,
    is_mask: bool = False,
) -> sitk.Image:
    """Resample an image or mask into the reference image space."""

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(moving_image)


def create_full_image_mask(image: sitk.Image, *, label: int = 1) -> sitk.Image:
    """Create a whole-image mask with the same geometry as ``image``."""

    mask_array = np.full(sitk.GetArrayFromImage(image).shape, label, dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    return mask


def n4_bias_field_correction(
    image: sitk.Image,
    *,
    shrink_factor: int = 4,
) -> sitk.Image:
    """Apply N4 bias-field correction to an intensity image."""

    shrink = [shrink_factor] * image.GetDimension()
    shrinked_image = sitk.Shrink(image, shrink)
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.UseMaskLabelOff()
    bias_field_filter.Execute(shrinked_image)
    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_bias_field)


def preprocess_image(image: sitk.Image, config: PreprocessingConfig) -> sitk.Image:
    """Apply configured preprocessing before radiomics extraction."""

    processed = image
    if config.cast_float32:
        processed = sitk.Cast(processed, sitk.sitkFloat32)
    if config.n4_bias_correction:
        processed = n4_bias_field_correction(
            processed,
            shrink_factor=config.n4_shrink_factor,
        )
    if config.denoise:
        processed = sitk.CurvatureAnisotropicDiffusion(
            processed,
            timeStep=config.denoise_time_step,
        )
    return processed
