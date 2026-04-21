from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

NIFTI_DTYPES: dict[int, np.dtype] = {
    2: np.dtype(np.uint8),
    4: np.dtype(np.int16),
    8: np.dtype(np.int32),
    16: np.dtype(np.float32),
    64: np.dtype(np.float64),
    256: np.dtype(np.int8),
    512: np.dtype(np.uint16),
    768: np.dtype(np.uint32),
    1024: np.dtype(np.int64),
    1280: np.dtype(np.uint64),
}


def _read_all_bytes(path: Path) -> bytes:
    if path.name.lower().endswith(".nii.gz"):
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _resolve_endianness(header: bytes) -> str:
    little = struct.unpack("<I", header[:4])[0]
    if little == 348:
        return "<"

    big = struct.unpack(">I", header[:4])[0]
    if big == 348:
        return ">"

    raise ValueError("Unsupported NIfTI header. Expected sizeof_hdr = 348.")


def load_nifti_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = _read_all_bytes(path)
    if len(raw) < 352:
        raise ValueError("Uploaded NIfTI file is too small to contain a valid volume.")

    header = raw[:348]
    endian = _resolve_endianness(header)

    dims = struct.unpack(f"{endian}8h", header[40:56])
    ndim = int(dims[0])
    if ndim < 2:
        raise ValueError("NIfTI volume must contain at least two dimensions.")

    shape = [int(size) for size in dims[1 : ndim + 1] if int(size) > 0]
    if not shape:
        raise ValueError("NIfTI volume does not expose a usable shape.")

    datatype_code = struct.unpack(f"{endian}h", header[70:72])[0]
    dtype = NIFTI_DTYPES.get(int(datatype_code))
    if dtype is None:
        raise ValueError(f"NIfTI datatype {datatype_code} is not supported in this build.")

    vox_offset = int(round(struct.unpack(f"{endian}f", header[108:112])[0]))
    slope = float(struct.unpack(f"{endian}f", header[112:116])[0])
    intercept = float(struct.unpack(f"{endian}f", header[116:120])[0])

    dtype = dtype.newbyteorder(endian)
    expected_values = int(np.prod(shape))
    volume = np.frombuffer(raw, dtype=dtype, offset=vox_offset)
    if volume.size < expected_values:
        raise ValueError("NIfTI file ended before the expected voxel data was available.")

    volume = volume[:expected_values].reshape(tuple(shape), order="F").astype(np.float32)
    if slope not in (0.0, 1.0):
        volume = volume * slope
    if intercept != 0.0:
        volume = volume + intercept

    while volume.ndim > 3:
        volume = volume[..., 0]

    return volume


def extract_middle_slice(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume
    if volume.ndim != 3:
        raise ValueError("Expected a 2D image or 3D NIfTI volume.")
    return volume[:, :, volume.shape[2] // 2]
