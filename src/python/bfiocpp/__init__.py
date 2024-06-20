from .tsreader import TSReader, Seq, FileType, get_ome_xml  # NOQA: F401
from .tswriter import TSWriter
from . import _version

__version__ = _version.get_versions()["version"]
