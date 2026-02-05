from bfiocpp import TSReader, TSWriter, Seq, FileType
import unittest
import requests, pathlib, shutil, logging, sys
# SEE : Initialization of bio-formats java backend https://bio-formats.readthedocs.io/en/stable/developers/java-library.html
# The order of initialization between ome_zarr.utils and bfio matters
from ome_zarr.utils import download as zarr_download
import bfio
import numpy as np
import tempfile, os

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
            FileType.OmeZarrV2,
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
                FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
                FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
                FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
                FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
                FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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
            FileType.OmeZarrV2,
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


class TestZarrV3Write(unittest.TestCase):
    """Tests for Zarr v3 write support"""

    def test_write_zarr_v3_basic(self):
        """Test basic Zarr v3 write and read-back"""
        with tempfile.TemporaryDirectory() as dir:
            test_file_path = os.path.join(dir, 'test_v3.zarr')

            # Create test data
            shape = [1, 1, 1, 100, 100]
            chunk_shape = [1, 1, 1, 64, 64]
            test_data = np.arange(100 * 100, dtype=np.uint16).reshape(shape)

            # Write using v3 format
            bw = TSWriter(test_file_path, shape, chunk_shape, "uint16", "TCZYX", FileType.OmeZarrV3)
            rows = Seq(0, 99, 1)
            cols = Seq(0, 99, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            bw.write_image_data(test_data, rows, cols, layers, channels, tsteps)
            bw.close()

            # Verify zarr.json exists (v3 format indicator)
            zarr_json_path = os.path.join(test_file_path, 'zarr.json')
            self.assertTrue(os.path.exists(zarr_json_path), "zarr.json should exist for v3 format")

            # Verify .zarray does NOT exist (v2 format indicator)
            zarray_path = os.path.join(test_file_path, '.zarray')
            self.assertFalse(os.path.exists(zarray_path), ".zarray should not exist for v3 format")

            # Read back using v3 reader
            br = TSReader(test_file_path, FileType.OmeZarrV3, "TCZYX")
            read_data = br.data(rows, cols, layers, channels, tsteps)

            # Verify data integrity
            self.assertEqual(read_data.shape, tuple(shape))
            self.assertEqual(read_data.dtype, np.uint16)
            self.assertTrue(np.array_equal(read_data, test_data))

    def test_write_zarr_v3_multiple_dtypes(self):
        """Test Zarr v3 write with different data types"""
        dtypes_to_test = [
            ("uint8", np.uint8),
            ("uint16", np.uint16),
            ("uint32", np.uint32),
            ("float32", np.float32),
        ]

        with tempfile.TemporaryDirectory() as dir:
            for dtype_str, np_dtype in dtypes_to_test:
                test_file_path = os.path.join(dir, f'test_v3_{dtype_str}.zarr')

                shape = [1, 1, 1, 50, 50]
                chunk_shape = [1, 1, 1, 32, 32]
                test_data = np.ones(shape, dtype=np_dtype) * 42

                bw = TSWriter(test_file_path, shape, chunk_shape, dtype_str, "TCZYX", FileType.OmeZarrV3)
                rows = Seq(0, 49, 1)
                cols = Seq(0, 49, 1)
                layers = Seq(0, 0, 1)
                channels = Seq(0, 0, 1)
                tsteps = Seq(0, 0, 1)
                bw.write_image_data(test_data, rows, cols, layers, channels, tsteps)
                bw.close()

                # Verify v3 format
                self.assertTrue(os.path.exists(os.path.join(test_file_path, 'zarr.json')),
                               f"zarr.json should exist for {dtype_str}")

                # Read back and verify
                br = TSReader(test_file_path, FileType.OmeZarrV3, "TCZYX")
                read_data = br.data(rows, cols, layers, channels, tsteps)
                self.assertEqual(read_data.dtype, np_dtype, f"dtype mismatch for {dtype_str}")
                self.assertTrue(np.allclose(read_data, test_data), f"data mismatch for {dtype_str}")

    def test_write_zarr_v3_chunked(self):
        """Test Zarr v3 write with chunked writes"""
        with tempfile.TemporaryDirectory() as dir:
            test_file_path = os.path.join(dir, 'test_v3_chunked.zarr')

            shape = [1, 1, 1, 200, 200]
            chunk_shape = [1, 1, 1, 100, 100]

            bw = TSWriter(test_file_path, shape, chunk_shape, "uint16", "TCZYX", FileType.OmeZarrV3)

            # Write in 4 chunks (2x2 grid)
            for y_start in [0, 100]:
                for x_start in [0, 100]:
                    chunk_data = np.full([1, 1, 1, 100, 100],
                                        fill_value=(y_start + x_start), dtype=np.uint16)
                    rows = Seq(y_start, y_start + 99, 1)
                    cols = Seq(x_start, x_start + 99, 1)
                    layers = Seq(0, 0, 1)
                    channels = Seq(0, 0, 1)
                    tsteps = Seq(0, 0, 1)
                    bw.write_image_data(chunk_data, rows, cols, layers, channels, tsteps)

            bw.close()

            # Read back full image
            br = TSReader(test_file_path, FileType.OmeZarrV3, "TCZYX")
            rows = Seq(0, 199, 1)
            cols = Seq(0, 199, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            read_data = br.data(rows, cols, layers, channels, tsteps)

            # Verify each quadrant has correct value
            self.assertTrue(np.all(read_data[0, 0, 0, :100, :100] == 0))    # top-left
            self.assertTrue(np.all(read_data[0, 0, 0, :100, 100:] == 100))  # top-right
            self.assertTrue(np.all(read_data[0, 0, 0, 100:, :100] == 100))  # bottom-left
            self.assertTrue(np.all(read_data[0, 0, 0, 100:, 100:] == 200))  # bottom-right

    def test_write_zarr_v2_default(self):
        """Test that default write (no FileType) creates v2 format"""
        with tempfile.TemporaryDirectory() as dir:
            test_file_path = os.path.join(dir, 'test_v2_default.zarr')

            shape = [1, 1, 1, 50, 50]
            chunk_shape = [1, 1, 1, 32, 32]
            test_data = np.ones(shape, dtype=np.uint16)

            # Write without specifying FileType (should default to v2)
            bw = TSWriter(test_file_path, shape, chunk_shape, "uint16", "TCZYX")
            rows = Seq(0, 49, 1)
            cols = Seq(0, 49, 1)
            layers = Seq(0, 0, 1)
            channels = Seq(0, 0, 1)
            tsteps = Seq(0, 0, 1)
            bw.write_image_data(test_data, rows, cols, layers, channels, tsteps)
            bw.close()

            # Verify .zarray exists (v2 format)
            self.assertTrue(os.path.exists(os.path.join(test_file_path, '.zarray')),
                           ".zarray should exist for default v2 format")
            # Verify zarr.json does NOT exist
            self.assertFalse(os.path.exists(os.path.join(test_file_path, 'zarr.json')),
                            "zarr.json should not exist for v2 format")