"""Re-export all dataset loader classes.

Each import is wrapped in a try/except so that loaders not yet implemented
(empty stubs or missing optional dependencies) do not prevent the rest of
the package from being imported.
"""

try:
    from src.data.loaders.alpaca import AlpacaLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.bipia import BipiaLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.dolly import DollyLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.hackaprompt import HackapromptLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.ifeval import IFEvalLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.injecagent import InjecAgentLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.lmsys_chat import LmsysChatLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.natural_instructions import NaturalInstructionsLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.notinject import NotInjectLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.open_prompt_injection import OpenPromptInjectionLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.spp import SppLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.struq import StruqLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.ultrachat import UltrachatLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

try:
    from src.data.loaders.wildguard import WildguardLoader
except (ImportError, Exception):
    pass  # type: ignore[assignment]

__all__ = [
    "AlpacaLoader",
    "BipiaLoader",
    "DollyLoader",
    "HackapromptLoader",
    "IFEvalLoader",
    "InjecAgentLoader",
    "LmsysChatLoader",
    "NaturalInstructionsLoader",
    "NotInjectLoader",
    "OpenPromptInjectionLoader",
    "SppLoader",
    "StruqLoader",
    "UltrachatLoader",
    "WildguardLoader",
]
