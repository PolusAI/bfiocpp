"""Round-trip tests for NIfTI-1 writer support in bfiocpp."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from bfiocpp import NiftiReader, NiftiWriter
from bfiocpp.libbfiocpp import Seq


class TestNiftiWrite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="bfiocpp_nifti_write_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _out(self, name):
        return os.path.join(self._tmpdir, name)

    # ------------------------------------------------------------------
    # Helper: read the full volume back and return it as (Z, Y, X) or
    # (T, Z, Y, X) shaped array.
    # ------------------------------------------------------------------
    def _readback(self, path, shape_zyx):
        """Read entire volume; returns array shaped (Z, Y, X)."""
        with NiftiReader(path) as r:
            nx, ny, nz = r.X, r.Y, r.Z
            arr = r.data(
                rows=Seq(0, ny - 1, 1),
                cols=Seq(0, nx - 1, 1),
                layers=Seq(0, nz - 1, 1),
            )
        # arr shape: (T=1, C=1, Z, Y, X)
        return arr[0, 0]

    # ------------------------------------------------------------------
    # test_write_read_uint16_3d
    # ------------------------------------------------------------------
    def test_write_read_uint16_3d(self):
        """Write a ZYX uint16 volume and read it back."""
        arr = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
        path = self._out("u16_3d.nii")

        with NiftiWriter(path, [4, 5, 6], np.dtype("uint16"), "ZYX") as w:
            w.write_image_data(
                arr,
                rows=Seq(0, 4, 1),
                cols=Seq(0, 5, 1),
                layers=Seq(0, 3, 1),
            )

        result = self._readback(path, arr.shape)
        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(result, arr)

    # ------------------------------------------------------------------
    # test_write_read_float32_3d
    # ------------------------------------------------------------------
    def test_write_read_float32_3d(self):
        """Write a ZYX float32 volume and read it back."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 4, 5)).astype(np.float32)
        path = self._out("f32_3d.nii")

        with NiftiWriter(path, [3, 4, 5], np.dtype("float32"), "ZYX") as w:
            w.write_image_data(
                arr,
                rows=Seq(0, 3, 1),
                cols=Seq(0, 4, 1),
                layers=Seq(0, 2, 1),
            )

        result = self._readback(path, arr.shape)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

    # ------------------------------------------------------------------
    # test_write_read_gz
    # ------------------------------------------------------------------
    def test_write_read_gz(self):
        """Write a .nii.gz file and read it back."""
        arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        path = self._out("gz_test.nii.gz")

        with NiftiWriter(path, [2, 3, 4], np.dtype("uint16"), "ZYX") as w:
            w.write_image_data(
                arr,
                rows=Seq(0, 2, 1),
                cols=Seq(0, 3, 1),
                layers=Seq(0, 1, 1),
            )

        result = self._readback(path, arr.shape)
        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(result, arr)

    # ------------------------------------------------------------------
    # test_write_subregion
    # ------------------------------------------------------------------
    def test_write_subregion(self):
        """Two separate partial WriteImageData calls combine correctly."""
        # Full image: Z=4, Y=5, X=6, uint16
        full = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
        path = self._out("subregion.nii")

        with NiftiWriter(path, [4, 5, 6], np.dtype("uint16"), "ZYX") as w:
            # Write first two z-slices
            w.write_image_data(
                full[:2],
                rows=Seq(0, 4, 1),
                cols=Seq(0, 5, 1),
                layers=Seq(0, 1, 1),
            )
            # Write last two z-slices
            w.write_image_data(
                full[2:],
                rows=Seq(0, 4, 1),
                cols=Seq(0, 5, 1),
                layers=Seq(2, 3, 1),
            )

        result = self._readback(path, full.shape)
        np.testing.assert_array_equal(result, full)

    # ------------------------------------------------------------------
    # test_write_read_4d
    # ------------------------------------------------------------------
    def test_write_read_4d(self):
        """Write two time-steps of a TZYX volume and read back."""
        # shape: T=2, Z=3, Y=4, X=5
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape(2, 3, 4, 5)
        path = self._out("4d_tzyx.nii")

        with NiftiWriter(path, [2, 3, 4, 5], np.dtype("int32"), "TZYX") as w:
            for t in range(2):
                w.write_image_data(
                    arr[t],
                    rows=Seq(0, 3, 1),
                    cols=Seq(0, 4, 1),
                    layers=Seq(0, 2, 1),
                    tsteps=Seq(t, t, 1),
                )

        # Read back both time steps
        with NiftiReader(path) as r:
            nx, ny, nz, nt = r.X, r.Y, r.Z, r.T
            self.assertEqual(nt, 2)
            for t in range(2):
                data = r.data(
                    rows=Seq(0, ny - 1, 1),
                    cols=Seq(0, nx - 1, 1),
                    layers=Seq(0, nz - 1, 1),
                    tsteps=Seq(t, t, 1),
                )
                # data shape: (T=1, C=1, Z, Y, X)
                np.testing.assert_array_equal(data[0, 0], arr[t])

    # ------------------------------------------------------------------
    # test_write_subregion_xy
    # ------------------------------------------------------------------
    def test_write_subregion_xy(self):
        """Partial writes in X/Y produce the correct result."""
        full = np.zeros((4, 6), dtype=np.uint16)
        full[1:3, 2:5] = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.uint16)
        path = self._out("subregion_xy.nii")

        with NiftiWriter(path, [4, 6], np.dtype("uint16"), "YX") as w:
            # Write just the interior patch
            patch = full[1:3, 2:5]
            w.write_image_data(
                patch,
                rows=Seq(1, 2, 1),
                cols=Seq(2, 4, 1),
            )

        result = self._readback(path, (1, 4, 6))
        # Z=1 slice; compare (Y, X) grid
        np.testing.assert_array_equal(result[0], full)

    # ------------------------------------------------------------------
    # test_invalid_dtype
    # ------------------------------------------------------------------
    def test_invalid_dtype(self):
        """Unknown dtype string raises RuntimeError."""
        with self.assertRaises(Exception):
            NiftiWriter(self._out("bad_dtype.nii"), [4, 5], np.dtype("complex64"), "YX")

    # ------------------------------------------------------------------
    # test_invalid_dimension_order_missing_x
    # ------------------------------------------------------------------
    def test_invalid_dimension_order_missing_x(self):
        """dimension_order without 'X' raises an error."""
        with self.assertRaises(Exception):
            NiftiWriter(self._out("bad_order.nii"), [4, 5], np.dtype("uint16"), "YZ")

    # ------------------------------------------------------------------
    # test_invalid_dimension_order_missing_y
    # ------------------------------------------------------------------
    def test_invalid_dimension_order_missing_y(self):
        """dimension_order without 'Y' raises an error."""
        with self.assertRaises(Exception):
            NiftiWriter(self._out("bad_order2.nii"), [4, 5], np.dtype("uint16"), "XZ")

    # ------------------------------------------------------------------
    # test_context_manager
    # ------------------------------------------------------------------
    def test_context_manager_gz(self):
        """Context manager correctly flushes a .nii.gz file."""
        arr = np.ones((2, 3, 4), dtype=np.float32) * 3.14
        path = self._out("ctx_mgr.nii.gz")

        with NiftiWriter(path, [2, 3, 4], np.dtype("float32"), "ZYX") as w:
            w.write_image_data(arr, Seq(0, 2, 1), Seq(0, 3, 1), Seq(0, 1, 1))

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

        result = self._readback(path, arr.shape)
        np.testing.assert_array_almost_equal(result, arr, decimal=5)


if __name__ == "__main__":
    unittest.main()
