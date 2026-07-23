"""Column contract for the four-value metadata tensor.

Only duration and slice type are encoded and fused with the signal features.
RMS pressure and mean velocity occupy the first two columns solely as
normalized auxiliary targets and evaluation labels.
"""

from __future__ import annotations


AUXILIARY_TARGET_FIELDS = ("rms_pa_db", "mean_vel")
MODEL_METADATA_FIELDS = ("duration_s", "slice_type")
METADATA_TENSOR_FIELDS = AUXILIARY_TARGET_FIELDS + MODEL_METADATA_FIELDS

AUXILIARY_TARGET_SLICE = slice(0, 2)
MODEL_METADATA_SLICE = slice(2, 4)
DURATION_COLUMN = 2
SLICE_TYPE_COLUMN = 3
