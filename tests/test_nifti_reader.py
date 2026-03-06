"""Tests for NIfTI-1 reader support in bfiocpp."""

import gzip
import os
import struct
import tempfile
import unittest

import numpy as np

from bfiocpp import NiftiReader
from bfiocpp.libbfiocpp import Seq


# ---------------------------------------------------------------------------
# Helpers to build minimal NIFTI-1 files
# ---------------------------------------------------------------------------

def _make_nifti1_header(
    nx, ny, nz, nt=1,
    datatype=512,    # uint16 by default
    vox_offset=352.0,
    scl_slope=0.0,
    scl_inter=0.0,
    pixdim=None,
):
    """Return 348-byte NIFTI-1 header as bytes."""
    if pixdim is None:
        pixdim = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    # Determine ndim and dim array
    if nt > 1:
        ndim = 4
    elif nz > 1:
        ndim = 3
    elif ny > 1:
        ndim = 2
    else:
        ndim = 1

    dims = [ndim, nx, ny, nz, nt, 1, 1, 1]

    # Map datatype to bitpix
    bitpix_map = {2: 8, 4: 16, 8: 32, 16: 32, 64: 64,
                  256: 8, 512: 16, 768: 32, 1024: 64, 1280: 64}
    bitpix = bitpix_map.get(datatype, 16)

    # Pack header fields (all little-endian)
    # Use 'h' for regular+dim_info pair: low byte = regular='r', high byte = dim_info=0
    header = struct.pack(
        '<i10s18sihh',
        348,          # sizeof_hdr (int32, offset 0)
        b'\x00' * 10, # data_type (offset 4)
        b'\x00' * 18, # db_name (offset 14)
        0,            # extents (int32, offset 32)
        0,            # session_error (int16, offset 36)
        ord('r'),     # regular (low byte) + dim_info=0 (high byte) packed as int16 at offset 38
    )
    # dim[8] (int16 x 8) — starts at offset 40
    header += struct.pack('<8h', *dims)
    # intent_p1, intent_p2, intent_p3, intent_code (3f + h)
    header += struct.pack('<3fh', 0.0, 0.0, 0.0, 0)
    # datatype, bitpix, slice_start
    header += struct.pack('<3h', datatype, bitpix, 0)
    # pixdim[8] (float x 8)
    header += struct.pack('<8f', *pixdim)
    # vox_offset, scl_slope, scl_inter
    header += struct.pack('<3f', vox_offset, scl_slope, scl_inter)
    # slice_end (int16), slice_code (char), xyzt_units (char)
    header += struct.pack('<h2B', 0, 0, 0)
    # cal_max, cal_min, slice_duration, toffset
    header += struct.pack('<4f', 0.0, 0.0, 0.0, 0.0)
    # glmax, glmin
    header += struct.pack('<2i', 0, 0)
    # descrip[80], aux_file[24]
    header += b'\x00' * 80 + b'\x00' * 24
    # qform_code, sform_code (int16 x 2)
    header += struct.pack('<2h', 0, 0)
    # quatern_b/c/d, qoffset_x/y/z (float x 6)
    header += struct.pack('<6f', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # srow_x[4], srow_y[4], srow_z[4] (float x 12)
    header += struct.pack('<12f', *([0.0] * 12))
    # intent_name[16]
    header += b'\x00' * 16
    # magic[4]: "n+1\0" for single-file NIfTI
    header += b'n+1\x00'

    assert len(header) == 348, f"Header length = {len(header)}, expected 348"
    return header


def _nifti_pixel_data(array: np.ndarray) -> bytes:
    """Return raw bytes of a numpy array in C order (x-fastest in NIFTI convention)."""
    # NIFTI stores x-fastest, which maps to column-major in the XY plane.
    # For a 3-D array indexed [z, y, x] this is just C-contiguous bytes.
    return array.tobytes(order='C')


def _make_nifti1_file(array: np.ndarray, datatype: int, tmp_dir: str,
                      fname="test.nii", scl_slope=0.0, scl_inter=0.0,
                      pixdim=None):
    """Write a minimal NIFTI-1 .nii file and return the path."""
    shape = array.shape  # (nz, ny, nx) or (nt, nz, ny, nx)
    if array.ndim == 3:
        nz, ny, nx = shape
        nt = 1
    elif array.ndim == 4:
        nt, nz, ny, nx = shape
    else:
        raise ValueError("Only 3-D and 4-D arrays supported")

    vox_offset = 352.0  # 348-byte header + 4-byte extension block
    header = _make_nifti1_header(
        nx=nx, ny=ny, nz=nz, nt=nt,
        datatype=datatype,
        vox_offset=vox_offset,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        pixdim=pixdim,
    )
    pixel_data = _nifti_pixel_data(array)

    path = os.path.join(tmp_dir, fname)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(b'\x00' * 4)  # 4-byte extension block (all zeros = no extension)
        f.write(pixel_data)
    return path


def _make_nifti1_gz_file(array: np.ndarray, datatype: int, tmp_dir: str,
                         fname="test.nii.gz", scl_slope=0.0, scl_inter=0.0,
                         pixdim=None):
    """Write a NIFTI-1 .nii.gz file and return the path."""
    nii_path = _make_nifti1_file(array, datatype, tmp_dir,
                                  fname=fname.replace('.gz', ''),
                                  scl_slope=scl_slope, scl_inter=scl_inter,
                                  pixdim=pixdim)
    gz_path = os.path.join(tmp_dir, fname)
    with open(nii_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(nii_path)
    return gz_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNiftiRead(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="bfiocpp_nifti_test_")

        # 3-D uint16 image: shape (nz=4, ny=5, nx=6)
        cls.arr_u16 = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
        cls.nii_u16 = _make_nifti1_file(
            cls.arr_u16, datatype=512, tmp_dir=cls._tmpdir, fname="u16.nii"
        )

        # 3-D float32 image
        cls.arr_f32 = np.random.rand(3, 4, 5).astype(np.float32)
        cls.nii_f32 = _make_nifti1_file(
            cls.arr_f32, datatype=16, tmp_dir=cls._tmpdir, fname="f32.nii"
        )

        # 3-D uint16 image with scaling: output must be float64
        cls.arr_scale_raw = np.ones((2, 3, 4), dtype=np.uint16) * 10
        cls.nii_scaled = _make_nifti1_file(
            cls.arr_scale_raw, datatype=512, tmp_dir=cls._tmpdir,
            fname="scaled.nii", scl_slope=2.0, scl_inter=1.0
        )

        # .nii.gz version of the uint16 image
        cls.arr_gz = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        cls.nii_gz = _make_nifti1_gz_file(
            cls.arr_gz, datatype=512, tmp_dir=cls._tmpdir, fname="gz.nii.gz"
        )

        # 3-D image with physical size info
        cls.arr_phys = np.zeros((2, 3, 4), dtype=np.float32)
        cls.nii_phys = _make_nifti1_file(
            cls.arr_phys, datatype=16, tmp_dir=cls._tmpdir, fname="phys.nii",
            pixdim=[1.0, 0.5, 0.25, 2.0, 0.0, 0.0, 0.0, 0.0],
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Shape / dimension tests
    # ------------------------------------------------------------------

    def test_shape_3d_uint16(self):
        with NiftiReader(self.nii_u16) as r:
            self.assertEqual(r.X, 6)   # nx
            self.assertEqual(r.Y, 5)   # ny
            self.assertEqual(r.Z, 4)   # nz
            self.assertEqual(r.C, 1)
            self.assertEqual(r.T, 1)

    def test_shape_3d_float32(self):
        with NiftiReader(self.nii_f32) as r:
            self.assertEqual(r.X, 5)
            self.assertEqual(r.Y, 4)
            self.assertEqual(r.Z, 3)

    def test_tile_dims_equal_image_dims(self):
        with NiftiReader(self.nii_u16) as r:
            self.assertEqual(r._reader.get_tile_height(), r.Y)
            self.assertEqual(r._reader.get_tile_width(), r.X)
            self.assertEqual(r._reader.get_tile_depth(), r.Z)

    # ------------------------------------------------------------------
    # Pixel value tests
    # ------------------------------------------------------------------

    def test_pixel_values_uint16(self):
        """Read entire volume and compare to original array."""
        with NiftiReader(self.nii_u16) as r:
            arr = r.data(
                rows=Seq(0, r.Y - 1, 1),
                cols=Seq(0, r.X - 1, 1),
                layers=Seq(0, r.Z - 1, 1),
            )
        # arr shape: (T, C, Z, Y, X)
        self.assertEqual(arr.shape, (1, 1, 4, 5, 6))
        self.assertEqual(arr.dtype, np.uint16)
        # Compare against original (z, y, x) → (T, C, Z, Y, X)
        np.testing.assert_array_equal(arr[0, 0], self.arr_u16)

    def test_pixel_values_float32(self):
        with NiftiReader(self.nii_f32) as r:
            arr = r.data(
                rows=Seq(0, r.Y - 1, 1),
                cols=Seq(0, r.X - 1, 1),
                layers=Seq(0, r.Z - 1, 1),
            )
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_array_almost_equal(arr[0, 0], self.arr_f32, decimal=6)

    def test_subregion_read(self):
        """Read a sub-region and check values match the slice."""
        with NiftiReader(self.nii_u16) as r:
            arr = r.data(
                rows=Seq(1, 3, 1),   # y 1..3
                cols=Seq(2, 4, 1),   # x 2..4
                layers=Seq(0, 1, 1), # z 0..1
            )
        self.assertEqual(arr.shape, (1, 1, 2, 3, 3))
        expected = self.arr_u16[0:2, 1:4, 2:5]
        np.testing.assert_array_equal(arr[0, 0], expected)

    # ------------------------------------------------------------------
    # Scaling test
    # ------------------------------------------------------------------

    def test_scaling(self):
        """scl_slope=2, scl_inter=1 → output float64 with val*2+1."""
        with NiftiReader(self.nii_scaled) as r:
            self.assertEqual(r.datatype, "double")
            arr = r.data(
                rows=Seq(0, r.Y - 1, 1),
                cols=Seq(0, r.X - 1, 1),
                layers=Seq(0, r.Z - 1, 1),
            )
        self.assertEqual(arr.dtype, np.float64)
        expected = self.arr_scale_raw.astype(np.float64) * 2.0 + 1.0
        np.testing.assert_array_almost_equal(arr[0, 0], expected, decimal=10)

    # ------------------------------------------------------------------
    # Compressed (.nii.gz) test
    # ------------------------------------------------------------------

    def test_gz_shape(self):
        with NiftiReader(self.nii_gz) as r:
            self.assertEqual(r.X, 4)
            self.assertEqual(r.Y, 3)
            self.assertEqual(r.Z, 2)

    def test_gz_pixel_values(self):
        with NiftiReader(self.nii_gz) as r:
            arr = r.data(
                rows=Seq(0, r.Y - 1, 1),
                cols=Seq(0, r.X - 1, 1),
                layers=Seq(0, r.Z - 1, 1),
            )
        self.assertEqual(arr.dtype, np.uint16)
        np.testing.assert_array_equal(arr[0, 0], self.arr_gz)

    # ------------------------------------------------------------------
    # Physical size test
    # ------------------------------------------------------------------

    def test_physical_size(self):
        with NiftiReader(self.nii_phys) as r:
            self.assertAlmostEqual(r.physical_size_x, 0.5, places=5)
            self.assertAlmostEqual(r.physical_size_y, 0.25, places=5)
            self.assertAlmostEqual(r.physical_size_z, 2.0, places=5)

    # ------------------------------------------------------------------
    # Datatype string tests
    # ------------------------------------------------------------------

    def test_datatype_uint16(self):
        with NiftiReader(self.nii_u16) as r:
            self.assertEqual(r.datatype, "uint16")

    def test_datatype_float32(self):
        with NiftiReader(self.nii_f32) as r:
            self.assertEqual(r.datatype, "float")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_invalid_extension(self):
        with self.assertRaises(RuntimeError):
            NiftiReader("/tmp/nofile.tiff")

    def test_nonexistent_file(self):
        with self.assertRaises(Exception):
            NiftiReader("/tmp/does_not_exist.nii")


if __name__ == "__main__":
    unittest.main()
