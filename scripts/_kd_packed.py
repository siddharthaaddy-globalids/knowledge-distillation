"""Exit 0 when the named directory holds a packed checkpoint kd will recognise.

    python scripts/_kd_packed.py /workspace/runs/<run-id>/quantized

Asked through kd.quantize.is_quantized rather than by looking for files here,
because the arena asks the same function (arena.py:897). A directory that looks
right to a shell script and wrong to is_quantized is precisely the failure this
exists to catch: the packing succeeds, the upload ships it, and the arena scores
four players instead of five with no error anywhere.
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        sys.exit("usage: _kd_packed.py <directory>")

    from kd import quantize

    return 0 if quantize.is_quantized(argv[0]) else 1


if __name__ == "__main__":
    sys.exit(main())
