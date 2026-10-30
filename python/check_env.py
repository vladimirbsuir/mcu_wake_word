#!/usr/bin/env python3
"""Small dependency probe used by the Tauri UI. Never imports heavy packages."""
from __future__ import annotations

import importlib.util
import json
import platform
import sys

PACKAGES = ["numpy", "soundfile", "librosa", "resampy", "tensorflow"]


def main() -> int:
    packages = {name: importlib.util.find_spec(name) is not None for name in PACKAGES}
    info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "all_packages_present": all(packages.values()),
        "supported_python": sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)),
    }
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
