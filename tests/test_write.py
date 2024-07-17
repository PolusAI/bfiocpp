from bfiocpp import TSReader, TSWriter, Seq, FileType
import unittest
import requests, pathlib, shutil, logging, sys
import bfio
import numpy as np
import tempfile, os
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "5025551.zarr": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
}

TEST_DIR = pathlib.Path(__file__).with_name("data")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.test")

if "-v" in sys.argv:
    logger.setLevel(logging.INFO)


def setUpModule():
    """Download images for testing"""
    TEST_DIR.mkdir(exist_ok=True)

    for file, url in TEST_IMAGES.items():
        logger.info(f"setup - Downloading: {file}")

        if not file.endswith(".zarr"):
            if TEST_DIR.joinpath(file).exists():
                continue

            r = requests.get(url)

            with open(TEST_DIR.joinpath(file), "wb") as fw:
                fw.write(r.content)
        else:
            if TEST_DIR.joinpath(file).exists():
                shutil.rmtree(TEST_DIR.joinpath(file))
            zarr_download(url, str(TEST_DIR))


def tearDownModule():
    """Remove test images"""

    logger.info("teardown - Removing test images...")
    shutil.rmtree(TEST_DIR)


class TestZarrWrite(unittest.TestCase):

    def test_write_zarr_5d(self):
        """test_write_zarr_5d - Write zarr using TSWriter"""

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

        with tempfile.TemporaryDirectory() as dir:
            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)

            bw = TSWriter(test_file_path, tmp.shape, tmp.shape, str(tmp.dtype), "TCZYX")
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)
            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 183750394
            assert tmp.shape == (1, 1, 1, 2700, 2702)

    def test_write_zarr_3d(self):
        """test_write_zarr_5d - Write zarr using TSWriter"""

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

        with tempfile.TemporaryDirectory() as dir:
            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)

            tmp =  np.expand_dims(np.squeeze(tmp), axis=0) # modify image to be 3D instead of 5D

            bw = TSWriter(test_file_path, tmp.shape, tmp.shape, str(tmp.dtype), "ZYX")
            bw.write_image_data(tmp, rows, cols, layers)
            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            tmp = br.data(rows, cols, layers)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 183750394
            assert tmp.shape == (1, 2700, 2702)

    def test_write_zarr_chunk_5d(self):
        """test_write_zarr_5d - Write zarr using TsWriter"""

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

        with tempfile.TemporaryDirectory() as dir:
            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            print(tmp.shape)
            chunk_shape = [1, 1, 1, 1350, 1351]
            #chunk_shape = tmp.shape
            print("dtype: " + str(tmp.dtype))
            bw = TSWriter(test_file_path, tmp.shape, chunk_shape, str(tmp.dtype), "TCZYX")

            rows = Seq(0, br._Y//2, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            tmp = br.data(rows, cols, layers, channels, tsteps)
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)

            rows = Seq(br._Y//2 , br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            tmp = br.data(rows, cols, layers, channels, tsteps)
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)

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

            print(tmp)

            assert tmp.dtype == np.uint8
            assert tmp.shape == (1, 1, 1, 2700, 2702)
            assert tmp.sum() == 183750394
    
    def test_write_zarr_3d(self):
        """test_write_zarr_5d - Write zarr using TsWriter"""

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

        # Update shape to (1, 3, 1, 2700, 2702)
        tmp = np.repeat(tmp, 3, axis=1)  

        with tempfile.TemporaryDirectory() as dir:
            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 2, 1)
            tsteps = Seq(0, 0, 1)

            bw = TSWriter(test_file_path, tmp.shape, tmp.shape, str(tmp.dtype), "TCZYX")
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)
            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 3*183750394
            assert tmp.shape == (1, 3, 1, 2700, 2702)


    def test_write_zarr_chunk_3d(self):
        """test_write_zarr_5d - Write zarr using TsWriter"""

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

        # Update shape to (1, 3, 1, 2700, 2702)
        tmp = np.repeat(tmp, 3, axis=1)  

        with tempfile.TemporaryDirectory() as dir:
            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            chunk_size = (1,1,1,2700,2702)

            bw = TSWriter(test_file_path, tmp.shape, chunk_size, str(tmp.dtype), "TCZYX")

            # write first channel
            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)

            # write second channel
            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(1, 1, 1)
            tsteps = Seq(0, 0, 1)
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)

            # write third channel
            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(2, 2, 1)
            tsteps = Seq(0, 0, 1)
            bw.write_image_data(tmp, rows, cols, layers, channels, tsteps)

            bw.close()

            br = TSReader(
                str(test_file_path),
                FileType.OmeZarr,
                "",
            )

            rows = Seq(0, br._Y - 1, 1)
            cols = Seq(0, br._X - 1, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 2, 1)
            tsteps = Seq(0, 0, 1)

            tmp = br.data(rows, cols, layers, channels, tsteps)

            assert tmp.dtype == np.uint8
            assert tmp.sum() == 3*183750394
            assert tmp.shape == (1, 3, 1, 2700, 2702)

    def test_invalid_dimension_order_no_X(self):

        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        with tempfile.TemporaryDirectory() as dir:

            test_file_path = os.path.join(dir, 'out/invalid.ome.zarr')
            chunk_size = (1,1,1,2700,2702)

            with self.assertRaises(Exception):
                TSWriter(test_file_path, tmp.shape, chunk_size, str(tmp.dtype), "TCZY")

    def test_invalid_dimension_order_no_Y(self):

        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        with tempfile.TemporaryDirectory() as dir:
            
            test_file_path = os.path.join(dir, 'out/invalid.ome.zarr')
            chunk_size = (1,1,1,2700,2702)

            with self.assertRaises(Exception):
                TSWriter(test_file_path, tmp.shape, chunk_size, str(tmp.dtype), "TCZX")

    def test_invalid_dimension_order_character(self):

        br = TSReader(
            str(TEST_DIR.joinpath("5025551.zarr/0")),
            FileType.OmeZarr,
            "",
        )

        rows = Seq(0, br._Y - 1, 1)
        cols = Seq(0, br._X - 1, 1)
        layers = Seq(0, 0, 1)
        channels = Seq(0, 0, 1)
        tsteps = Seq(0, 0, 1)
        tmp = br.data(rows, cols, layers, channels, tsteps)

        with tempfile.TemporaryDirectory() as dir:
            
            test_file_path = os.path.join(dir, 'out/invalid.ome.zarr')
            chunk_size = (1,1,1,2700,2702)

            with self.assertRaises(Exception):
                TSWriter(test_file_path, tmp.shape, chunk_size, str(tmp.dtype), "ATCZYX")