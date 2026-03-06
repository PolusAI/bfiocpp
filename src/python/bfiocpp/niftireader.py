import numpy as np
from .libbfiocpp import NiftiReaderCPP, Seq  # NOQA: F401


class NiftiReader:
    """Read NIfTI-1/2 files (.nii and .nii.gz)."""

    def __init__(self, file_name: str) -> None:
        self._reader = NiftiReaderCPP(file_name)
        self._X: int = self._reader.get_image_width()
        self._Y: int = self._reader.get_image_height()
        self._Z: int = self._reader.get_image_depth()
        self._C: int = self._reader.get_channel_count()
        self._T: int = self._reader.get_tstep_count()
        self._datatype: str = self._reader.get_datatype()

    @property
    def X(self) -> int:
        return self._X

    @property
    def Y(self) -> int:
        return self._Y

    @property
    def Z(self) -> int:
        return self._Z

    @property
    def C(self) -> int:
        return self._C

    @property
    def T(self) -> int:
        return self._T

    @property
    def datatype(self) -> str:
        return self._datatype

    @property
    def physical_size_x(self) -> float:
        return self._reader.get_physical_size_x()

    @property
    def physical_size_y(self) -> float:
        return self._reader.get_physical_size_y()

    @property
    def physical_size_z(self) -> float:
        return self._reader.get_physical_size_z()

    def data(
        self,
        rows: Seq,
        cols: Seq,
        layers: Seq = None,
        channels: Seq = None,
        tsteps: Seq = None,
    ) -> np.ndarray:
        if layers is None:
            layers = Seq(0, 0, 1)
        if channels is None:
            channels = Seq(0, 0, 1)
        if tsteps is None:
            tsteps = Seq(0, 0, 1)
        return self._reader.get_image_data(rows, cols, layers, channels, tsteps)

    def close(self) -> None:
        pass

    def __enter__(self) -> "NiftiReader":
        return self

    def __del__(self) -> None:
        self.close()

    def __exit__(self, type_class, value, traceback) -> None:
        self.close()
