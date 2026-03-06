import numpy as np
from .libbfiocpp import NiftiWriterCPP, Seq  # NOQA: F401


class NiftiWriter:
    """Write NIfTI-1 files (.nii and .nii.gz).

    Parameters
    ----------
    file_name : str
        Output path ending in ``.nii`` or ``.nii.gz``.
    image_shape : list[int]
        Full image shape matching *dimension_order*.
    dtype : np.dtype
        Element data type (e.g. ``np.dtype("uint16")``).
    dimension_order : str
        A combination of ``T``, ``C``, ``Z``, ``Y``, ``X``
        describing the axis order of *image_shape*.
        Must contain at least ``X`` and ``Y``.
        ``C`` is accepted but ignored by NIfTI.
    """

    def __init__(
        self,
        file_name: str,
        image_shape: list,
        dtype: np.dtype,
        dimension_order: str,
    ) -> None:
        self._writer = NiftiWriterCPP(
            file_name,
            [int(d) for d in image_shape],
            str(np.dtype(dtype)),
            dimension_order,
        )
        self._dtype = np.dtype(dtype)

    def write_image_data(
        self,
        image_data: np.ndarray,
        rows: Seq,
        cols: Seq,
        layers: Seq = None,
        channels: Seq = None,
        tsteps: Seq = None,
    ) -> None:
        if not isinstance(image_data, np.ndarray):
            raise ValueError("image_data must be a numpy ndarray")
        # Ensure correct dtype and contiguous C layout before passing to C++
        flat = np.ascontiguousarray(image_data, dtype=self._dtype).flatten()
        self._writer.write_image_data(flat, rows, cols, layers, channels, tsteps)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "NiftiWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
