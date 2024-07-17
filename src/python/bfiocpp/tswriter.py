import numpy as np
from typing import Optional
from .libbfiocpp import TsWriterCPP, Seq


class TSWriter:

    def __init__(
        self,
        file_name: str,
        image_shape: list,
        chunk_shape: list,
        dtype: np.dtype,
        dimension_order: str,
    ):
        """Initialize tensorstore Zarr writer

        file_name: Path to write file to
        """

        self._image_writer: TsWriterCPP = TsWriterCPP(
            file_name, image_shape, chunk_shape, str(dtype), dimension_order
        )

    def write_image_data(
        self,
        image_data: np.ndarray,
        rows: Seq,
        cols: Seq,
        layers: Optional[Seq] = None,
        channels: Optional[Seq] = None,
        tsteps: Optional[Seq] = None,
    ):
        """Write image data to file

        image_data: 5d numpy array containing image data
        """

        if not isinstance(image_data, np.ndarray):

            raise ValueError("Image data must be a 5d numpy array")

        try:
            self._image_writer.write_image_data(
                image_data.flatten(), rows, cols, layers, channels, tsteps
            )

        except Exception as e:
            raise RuntimeError(f"Error writing image data: {e.what}")

    def close(self):

        pass

    def __enter__(self) -> "TSWriter":
        """Handle entrance to a context manager.

        This code is called when a `with` statement is used.
            ...
        """

        return self

    def __del__(self) -> None:
        """Handle file deletion.

        This code runs when an object is deleted..
        """

        self.close()

    def __exit__(self, type_class, value, traceback) -> None:
        """Handle exit from the context manager.

        This code runs when exiting a `with` statement.
        """

        self.close()
