from bfiocpp import TSReader, Seq, FileType
import unittest
import bfio
import numpy as np
import random

from . import TEST_DIR

class TestOmeTiffRead(unittest.TestCase):

    def test_read_ome_tif_full(self):
        """test_read_ome_tif_full - Read tiff using TSTiffReader"""
        br = TSReader(
            str(TEST_DIR.joinpath("p01_x01_y01_wx0_wy0_c1.ome.tif")),
            FileType.OmeTiff,
            "",
        )
        assert br._X == 1080
        assert br._Y == 1080
        assert br._Z == 1
        assert br._C == 1
        assert br._T == 1

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, br._Z - 1, 1)
        channels = Seq(0, br._C - 1, 1)
        tsteps = Seq(0, br._T - 1, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        assert tmp.dtype == np.uint16
        assert tmp.sum() == 437949929
        assert tmp.shape == (1, 1, 1, 1080, 1080)

    def test_read_ome_tif_partial(self):
        """test_read_ome_tif_partial - Read partial tiff read"""
        with TSReader(
            str(TEST_DIR.joinpath("p01_x01_y01_wx0_wy0_c1.ome.tif")),
            FileType.OmeTiff,
            "",
        ) as br:
            rows = Seq(0, 1023, 1)
            cols = Seq(0, 1023, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint16
            assert tmp.sum() == 393970437
            assert tmp.shape == (1, 1, 1, 1024, 1024)

    def test_read_unaligned_tile_boundary(self):
        """test_read_unaligned_tile_boundary - Read partial tiff read without alinged tile boundary"""
        # create a 2D numpy array filled with random integer form 0-255
        img_height = 8000
        img_width = 7500
        source_data = np.random.randint(
            0, 256, (img_height, img_width), dtype=np.uint16
        )
        with bfio.BioWriter(
            str(TEST_DIR.joinpath("test_output.ome.tiff")),
            X=img_width,
            Y=img_height,
            dtype=np.uint16,
        ) as bw:
            bw[0:img_height, 0:img_width, 0, 0, 0] = source_data

        x_max = source_data.shape[0]
        y_max = source_data.shape[1]

        with TSReader(
            str(TEST_DIR.joinpath("test_output.ome.tiff")), FileType.OmeTiff, ""
        ) as test_br:
            for i in range(100):
                x_start = random.randint(0, x_max)
                y_start = random.randint(0, y_max)
                x_step = random.randint(1, 2 * 1024)
                y_step = random.randint(1, 3 * 1024)

                x_end = x_start + x_step
                y_end = y_start + y_step

                if x_end > x_max:
                    x_end = x_max

                if y_end > y_max:
                    y_end = y_max

                rows = Seq(x_start, x_end - 1, 1)
                cols = Seq(y_start, y_end - 1, 1)
                layers = Seq(0, 0, 1)
                channels = Seq(0, 0, 1)
                tsteps = Seq(0, 0, 1)
                test_data = test_br.data(rows, cols, layers, channels, tsteps)
                test_data = test_data.transpose(3, 4, 0, 1, 2)

                while test_data.shape[-1] == 1 and test_data.ndim > 2:
                    test_data = test_data[..., 0]

                assert (
                    np.sum(source_data[x_start:x_end, y_start:y_end] - test_data) == 0
                )

    def test_read_ome_tif_3d(self):
        pass

    def test_read_ome_tif_4d(self):
        """test_read_ome_tif_4d - Read 4D data"""
        br = TSReader(str(TEST_DIR.joinpath("4d_array.ome.tif")), FileType.OmeTiff, "")
        assert br._X == 672
        assert br._Y == 512
        assert br._Z == 21
        assert br._C == 3
        assert br._T == 1

        rows = Seq(0, 255, 1)
        cols = Seq(0, 127, 1)
        layers = Seq(1, 1, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)
        print(tmp.sum())
        assert tmp.sum() == 7898130

        rows = Seq(0, 255, 1)
        cols = Seq(0, 127, 1)
        layers = Seq(2, 2, 1)
        channels = Seq(1, 1, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)
        assert tmp.sum() == 7828625

        rows = Seq(0, 255, 1)
        cols = Seq(0, 127, 1)
        layers = Seq(15, 15, 1)
        channels = Seq(2, 2, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)
        assert tmp.sum() == 30206173

    def test_read_ome_tif_5d(self):
        pass


class TestOmeZarrRead(unittest.TestCase):

    def test_read_zarr_2d_slice(self):
        """test_read_zarr_2d_slice - Read tiff using TSReader"""
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

        assert tmp.dtype == np.uint8
        assert tmp.sum() == 183750394
        assert tmp.shape == (1, 1, 1, 2700, 2702)

    def test_read_zarr_4d_slice(self):
        """test_read_zarr_4d_slice - Read tiff using TSReader"""
        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )

        rows = Seq(0, 1023, 1)
        cols = Seq(0, 1023, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 3, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        assert tmp.dtype == np.uint8
        assert tmp.sum() == 81778531
        assert tmp.shape == (1, 4, 1, 1024, 1024)
