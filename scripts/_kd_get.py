"""Print one resolved config value, for shell scripts that must not guess.

    python scripts/_kd_get.py configs/enlibra/enlibraQ3-14B-score.yaml s3.cache_dir

A missing key prints nothing and exits 0: the caller decides whether absence is
fatal, and every one of them says so in its own words. Reading the value out of
the same loader kd uses - `extends` chain, defaults and all - is the point. The
alternative is a grep for a key in one file, which silently misses anything
inherited from _base.yaml.

KD_* ENV VARS ARE HONOURED, which they were not before, and the bug that fixes
is a nasty one. A pod script reads its paths through here and then hands the
same config to `kd`, which loads it with use_env=True. With the environment
ignored on this side only, `export KD_EVAL_ADAPTER=...` retargeted the python
half of a run and not the shell half: the script would check one adapter, print
it, and the command it launched would use another. Both halves now read the
same layered result.
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        sys.exit("usage: _kd_get.py <config> <dotted.key>")

    from kd.config import load_config

    value = load_config(argv[0], use_env=True)
    for key in argv[1].split("."):
        value = (value or {}).get(key)
        if value is None:
            break
    print("" if value is None else value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
