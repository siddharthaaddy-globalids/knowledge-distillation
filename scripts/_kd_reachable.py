"""Exit 0 when the bucket answers RIGHT NOW with the credentials in this shell.

    python scripts/_kd_reachable.py <config> [uri]

The uri defaults to the config's evaluation.adapter, which is in the same
bucket every step reads and writes.

WHAT THIS IS FOR. The credentials on these pods last fifteen minutes, and the
failure that costs the most is not a missing token - that one is obvious - but
a LIVE-LOOKING one that expired during the previous step. `AWS_ACCESS_KEY_ID`
is still set, so a presence check passes, and the failure surfaces partway
through an eight-gigabyte upload with the pod still billing.

One ListObjectsV2 capped at a single key turns that into a one-second answer
before the expensive thing starts. Asked through kd.remote.s3.reachable rather
than with a bare boto3 call so that the endpoint_url and region a config sets -
MinIO, R2, a RunPod volume - are the ones probed.

Prints nothing on success: the caller is a shell script that wants an exit code
and prints its own message.
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or len(argv) > 2:
        sys.exit("usage: _kd_reachable.py <config> [uri]")

    from kd.config import load_config
    from kd.remote import s3

    config = load_config(argv[0], use_env=True)
    uri = argv[1] if len(argv) > 1 else (config.get("evaluation") or {}).get("adapter")
    if not uri:
        sys.exit("no uri given and the config sets no evaluation.adapter")

    ok, detail = s3.reachable(config, uri)
    if ok:
        return 0
    print(detail, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
