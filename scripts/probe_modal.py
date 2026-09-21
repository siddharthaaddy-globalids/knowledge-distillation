#!/usr/bin/env python3
"""Which models on an OpenAI-compatible endpoint will these credentials actually run?

Listing /v1/models says what is SERVED. It does not say what your token is allowed
to call - a deployment can authorise per model, and the difference only shows up
on the first completion, 140 questions into a benchmark.

So ask each one a question that costs a token or two, and report what came back.

    python scripts/probe_modal.py
    python scripts/probe_modal.py --base-url https://... --api-key sk-...

Reads MODAL_BASE_URL, MODAL_API_KEY / MODAL_TOKEN, MODAL_KEY, MODAL_SECRET, the
same as modal_arena.py, and shares its client so the request is byte-for-byte
what the benchmark would send.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from modal_arena import (  # noqa: E402
    ModalClient, auth_headers, build_ssl_context, list_models, parse_header,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("MODAL_BASE_URL"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--modal-key", default=None)
    ap.add_argument("--modal-secret", default=None)
    ap.add_argument("--header", action="append", default=[], metavar="'Name: value'")
    ap.add_argument("--ca-bundle", default=None)
    ap.add_argument("--model", action="append", default=[],
                    help="probe only these (default: everything /v1/models lists)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    if not args.base_url:
        raise SystemExit("no endpoint: pass --base-url or set MODAL_BASE_URL")

    ssl_context, trust = build_ssl_context(args.ca_bundle)
    headers = auth_headers(
        args.api_key or os.environ.get("MODAL_API_KEY") or os.environ.get("MODAL_TOKEN"),
        args.modal_key or os.environ.get("MODAL_KEY"),
        args.modal_secret or os.environ.get("MODAL_SECRET"),
        dict(parse_header(h) for h in args.header),
    )
    sent = [k for k in ("Authorization", "Modal-Key", "Modal-Secret") if k in headers]
    print(f"endpoint   {args.base_url}")
    print(f"verifying  {trust}")
    print(f"sending    {', '.join(sent) or 'no credentials'}\n")

    models = args.model
    if not models:
        served, err = list_models(args.base_url, headers, args.timeout, ssl_context)
        if err:
            raise SystemExit(f"could not list models: {err}")
        models = [m.get("id") for m in served]
        if not models:
            raise SystemExit("the endpoint listed no models; name one with --model")

    width = max(len(m) for m in models)
    ok = []
    for model in models:
        client = ModalClient(
            headers=headers, ssl_context=ssl_context, api_key="", model=model,
            base_url=args.base_url, max_tokens=8, temperature=0.0,
            reasoning_effort=None, timeout=args.timeout, max_retries=0,
        )
        out = client.complete("Reply with the single letter A.", "")
        if out["error"]:
            print(f"  {model:{width}}  REFUSED   {out['error'][:110]}")
        else:
            ok.append(model)
            said = (out["completion"] or "").strip().replace("\n", " ")[:40] or "(empty)"
            print(f"  {model:{width}}  OK        {out['latency_s']}s, said {said!r}")

    print()
    if ok:
        print("usable with these credentials:")
        for model in ok:
            print(f"  --model {model}")
    else:
        print("none of them accepted these credentials. The token may be scoped to a\n"
              "different model, or a different token is needed for this endpoint.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
