try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

try:
    __version__ = version("chatterbox-pdzd")
except Exception:
    __version__ = "0.1.7+pdzd"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES