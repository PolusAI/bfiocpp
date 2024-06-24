import numpy as np
from .libbfiocpp import TsWriterCPP


class TSWriter:

    def __init__(self, file_name: str):
        """ Initialize tensorstore Zarr writer

        file_name: Path to write file to
        """

        self._image_writer: TsWriterCPP = TsWriterCPP(file_name)

    def write_image(self, image_data: np.ndarray, image_shape: list, chunk_shape: list):
        """ Write image data to file

        image_data: 5d numpy array containing image data
        """

        if (not isinstance(image_data, np.ndarray)):

            raise ValueError("Image data must be a 5d numpy array")

        try:
            self._image_writer.write(image_data.flatten(), image_shape, chunk_shape)
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
