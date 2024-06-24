import unittest
import tempfile
import os
import numpy as np
import pathlib

from bfiocpp import TSWriter, TSReader, Seq, FileType

TEST_DIR = pathlib.Path(__file__).with_name("data")

class TestZarrWrite(unittest.TestCase):

    def test_write_zarr_2d(self):
        """test_write_zarr_2d - Write zarr using TSWrtier"""

        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )
        assert br._X == 2702
        assert br._Y == 2700
        assert br._Z == 1
        assert br._C == 27
        assert br._T == 1

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        with tempfile.TemporaryDirectory() as test_dir:
            # Use the temporary directory
            test_file_path = os.path.join(test_dir, 'out/test.ome.zarr')

            bw = TSWriter(test_file_path)
            bw.write_image(tmp, tmp.shape, tmp.shape)
            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 183750394
            assert tmp.shape == (1, 1, 1, 2700, 2702)

