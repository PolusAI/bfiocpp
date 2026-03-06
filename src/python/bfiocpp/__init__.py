from .tsreader import TSReader, Seq, FileType, get_ome_xml  # NOQA: F401
from .tswriter import TSWriter  # NOQA: F401
from .niftireader import NiftiReader  # NOQA: F401
from .niftiwriter import NiftiWriter  # NOQA: F401
from . import _version

__version__ = _version.get_versions()["version"]
