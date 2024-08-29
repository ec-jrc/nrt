__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from importlib.metadata import version

try:
    __version__ = version("nrt")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"
