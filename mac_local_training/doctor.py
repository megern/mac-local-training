"""Inspect local capabilities without importing MLX or loading a model."""
from importlib.util import find_spec
import platform
import sys


def diagnose():
    compatible = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    modules = {name: find_spec(name) is not None for name in ('mlx', 'mlx_lm', 'psutil')}
    return {'python': sys.version.split()[0], 'system': platform.system(), 'machine': platform.machine(),
            'planning_available': True, 'training_platform_supported': compatible,
            'modules_present': modules,
            'training_candidate': compatible and all(modules.values()),
            'scope': 'Module discovery only; does not initialize Metal, test allocations, download weights or guarantee a model fits.'}
