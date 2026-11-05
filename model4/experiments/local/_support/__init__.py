"""Private support for the local research methods."""
import importlib.util
from pathlib import Path


def load_stack(name):
    """Create private mutable experiment state without changing import paths.

    Historical runners assign different output roots and model adapters on the
    stack module. Keep those assignments isolated, as in the original loaders.
    """
    spec = importlib.util.spec_from_file_location(f"{__name__}.{name}", Path(__file__).with_name("stack.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
