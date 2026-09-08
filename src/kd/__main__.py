"""Entry point for `python -m kd`.

The `kd` console script is the friendly spelling, but it depends on a generated
executable being on PATH and executable - which is not a given inside a container,
a CI step, or a Windows machine whose security software distrusts newly written
binaries. `python -m kd` needs none of that, so it is what the generated runners
call.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
