import requests, pathlib, shutil, logging, sys
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "5025551.zarr": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
    "p01_x01_y01_wx0_wy0_c1.ome.tif": "https://raw.githubusercontent.com/sameeul/polus-test-data/main/bfio/p01_x01_y01_wx0_wy0_c1.ome.tif",
    "Plate1-Blue-A-12-Scene-3-P3-F2-03.czi": "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi",
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

    """Load the czi image, and save as a npy file for further testing."""
    with bfio.BioReader(
        TEST_DIR.joinpath("Plate1-Blue-A-12-Scene-3-P3-F2-03.czi")
    ) as br:
        with bfio.BioWriter(
            TEST_DIR.joinpath("4d_array.ome.tif"),
            metadata=br.metadata,
            X=br.X,
            Y=br.Y,
            Z=br.Z,
            C=br.C,
            T=br.T,
        ) as bw:
            bw[:] = br[:]