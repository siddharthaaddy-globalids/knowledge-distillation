"""
One place where a LoRA adapter is added into its base model.

Four callers needed this and each grew its own copy: `kd arena` to score the
distilled player, `kd publish` to ship a merged repo, `scripts/merge.py` to talk
to one by hand, and now the vLLM engine, which cannot be handed an adapter at
all - it wants a dense checkpoint on disk.

The copies had drifted, and not harmlessly. `kd arena` trimmed the base's output
layer to the width the adapter was built against before merging; `kd publish`
did not, so publishing an adapter trained against a trimmed teacher died on a
shape mismatch that the arena had already solved two files away. That is the
whole argument for this module: the trap below has one correct answer, and there
should be one copy of it.

THE VOCABULARY TRAP
-------------------
Stock Qwen checkpoints pad `vocab_size` up to a multiple of 128 for tensor
alignment: 151936 rows against a tokenizer holding 151665 real tokens. A
checkpoint fine-tuned through `resize_token_embeddings(len(tokenizer))` has had
that padding trimmed - and `kd.train` trims the student to match such a teacher
before training, so the adapter is built against the trimmed width.

Hand that adapter to an untrimmed base and PEFT refuses the state dict. So the
base is trimmed here too, to the same width, taken from whichever of these can
answer first:

    the adapter's kd-meta.json      what kd.train recorded. The only route that
                                    needs nothing but the adapter directory,
                                    which is why it is first.
    the adapter's saved embedding   runs from before kd.train stopped saving one
    the training config             derived from the student and teacher configs
    the run bundle beside it        config.resolved.yaml names the teacher

and left alone when none of them apply, which is the ordinary case.

NARROWER ONLY, EVER
-------------------
`resize_token_embeddings` grows as readily as it shrinks, and growing appends
rows initialised from nothing - weights the adapter never saw and no token id
indexes. Training only ever trimmed (`paths.vocab_target` takes the MIN of the
two widths), so a route asking to WIDEN has returned a wrong answer rather than
an instruction to follow, and is refused with a warning instead of obeyed.
"""

import json
import os

# Written into a materialised directory beside the weights, naming what was
# merged to produce it. Without it a stale directory from a different adapter is
# indistinguishable from a fresh one, and `reuse` would hand back the wrong
# model - which reads as a bad training run rather than as a bug.
MERGE_STAMP = "kd-merged.json"


def _say(log, message):
    """Progress, to whatever the caller logs with. print() when it has none."""
    if log is not None:
        log.info(message)
    else:
        print(message)


def _bundle_vocab(adapter):
    """The teacher's width, from the run bundle the adapter was written into.

    kd.runlog puts config.resolved.yaml one level above final_adapter/, and that
    file names the teacher the student was trimmed to match. Absent whenever the
    adapter has been moved on its own, which is why this is the last route
    rather than the only one.
    """
    resolved = os.path.join(
        os.path.dirname(os.path.abspath(str(adapter))), "config.resolved.yaml")
    if not os.path.isfile(resolved):
        return None
    try:
        import yaml
        from transformers import AutoConfig

        with open(resolved, encoding="utf-8") as handle:
            teacher = ((yaml.safe_load(handle) or {}).get("models") or {}).get("teacher")
        if not teacher or not (os.path.isdir(str(teacher)) or "/" in str(teacher)):
            return None
        return int(AutoConfig.from_pretrained(teacher).vocab_size)
    except Exception:  # noqa: BLE001 - a missing file, a fetch, a parse
        return None


def vocab_width(adapter, config=None, tokenizer_length=None, explicit=None):
    """(the width to trim the base to, where that came from), or (None, None).

    The routes are tried most-authoritative first; see the module docstring for
    why they are in this order. A genuine vocabulary difference - one no trim
    can bridge - is raised by `paths.vocab_target` and deliberately NOT caught
    here, because falling back past it would merge two models that disagree
    about what a token id means.
    """
    from . import paths

    if explicit:
        return int(explicit), "the width you named"

    recorded = paths.adapter_meta(adapter).get("vocab_size")
    if recorded:
        return int(recorded), "the adapter's kd-meta.json"

    saved = paths.adapter_vocab_size(adapter)
    if saved:
        return int(saved), "the adapter's saved embedding"

    models = (config or {}).get("models") or {}
    student = paths.adapter_base(adapter) or models.get("student")
    if student and models.get("teacher") and tokenizer_length:
        try:
            derived = paths.vocab_target(student, models["teacher"],
                                         tokenizer_length)
        except (OSError, ImportError):
            derived = None      # the teacher is not reachable from here
        if derived:
            return int(derived), "the student and teacher configs"

    found = _bundle_vocab(adapter)
    if found:
        return int(found), "the run bundle's config.resolved.yaml"
    return None, None


def trim(model, target, source=None, label="model", log=None):
    """Trim `model`'s output layer to `target`. Never widens it; see the header.

    True when something was changed, which is what `materialise` records so a
    later reader can tell a trimmed checkpoint from an untouched one.
    """
    if not target:
        return False
    current = int(model.config.vocab_size)
    if int(target) > current:
        _say(log, f"      !! {source or 'the adapter'} says {target} but {label} "
                  f"is {current}; leaving it alone - widening would invent rows "
                  f"the adapter never saw")
        return False
    from . import paths
    return paths.fit_vocab(model, int(target), label=label)


def merge_adapter(adapter, base_id=None, config=None, tokenizer_length=None,
                  dtype=None, device=None, log=None, vocab=None):
    """Base + adapter, added together, as one ordinary model.

    `base_id` defaults to whatever the adapter records, which is what makes an
    adapter portable: the directory alone says what it goes on top of. `vocab`
    overrides every route in `vocab_width`, for the case where all of them are
    absent and a person knows the answer.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from . import paths

    adapter = str(adapter)
    base_id = base_id or paths.adapter_base(adapter)
    if not base_id:
        raise ValueError(
            f"{adapter} does not record the base model it was trained against, "
            f"so the base has to be named explicitly.")

    _say(log, f"      merging  {adapter}")
    _say(log, f"      into     {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, low_cpu_mem_usage=True)
    target, source = vocab_width(adapter, config=config,
                                 tokenizer_length=tokenizer_length,
                                 explicit=vocab)
    trim(model, target, source=source, label="student", log=log)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device) if device else model


def is_dense_checkpoint(where):
    """True when `where` is a directory transformers loads with no PEFT involved."""
    where = str(where or "")
    return bool(where) and os.path.isdir(where) \
        and os.path.isfile(os.path.join(where, "config.json")) \
        and not os.path.isfile(os.path.join(where, "adapter_config.json"))


def _stamp_matches(out_dir, base_id, adapter):
    """True when out_dir already holds exactly this merge, weights included."""
    path = os.path.join(out_dir, MERGE_STAMP)
    if not (os.path.isfile(path) and is_dense_checkpoint(out_dir)):
        return False
    if not any(name.endswith(".safetensors") for name in os.listdir(out_dir)):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            stamp = json.load(handle)
    except Exception:  # noqa: BLE001 - a truncated write is a reason to redo it
        return False
    return (stamp.get("base") == str(base_id)
            and stamp.get("adapter") == str(adapter))


def materialise(out_dir, model_id=None, adapter=None, config=None,
                tokenizer=None, dtype=None, log=None, reuse=True):
    """A dense checkpoint on disk for something that may only be base + adapter.

    Returns a path - or a hub id - that `AutoModelForCausalLM.from_pretrained`,
    and more to the point vLLM and the quantiser, can be pointed at with no PEFT
    anywhere in the picture.

    NOTHING IS WRITTEN WHEN NOTHING NEEDS TO BE. With no adapter, `model_id` is
    already exactly such a thing and comes straight back untouched, hub id or
    local directory. Only the base+adapter case costs a copy of the weights.

    `reuse` returns an existing directory when its stamp names this same base
    and adapter, so a `--from arena` rerun does not re-merge eight gigabytes it
    merged twenty minutes ago. A stamp naming anything else is overwritten: the
    alternative is scoring one adapter and reporting it as another.
    """
    if not adapter:
        if not model_id:
            raise ValueError("materialise needs a model_id, an adapter, or both")
        return str(model_id)

    from . import paths

    base_id = model_id or paths.adapter_base(adapter)
    out_dir = os.path.abspath(str(out_dir))
    if reuse and _stamp_matches(out_dir, base_id, adapter):
        _say(log, f"      merged already: {out_dir}")
        return out_dir

    model = merge_adapter(adapter, base_id, config=config,
                          tokenizer_length=len(tokenizer) if tokenizer else None,
                          dtype=dtype, log=log)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    # The tokenizer goes with it, always. A merged directory without one is
    # something nobody can load without knowing which tokenizer it wants - and
    # vLLM, handed the directory, would go looking for the base model's.
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, MERGE_STAMP), "w", encoding="utf-8") as handle:
        json.dump({"base": str(base_id), "adapter": str(adapter),
                   "vocab_size": int(model.config.vocab_size)}, handle, indent=2)

    size = sum(os.path.getsize(os.path.join(out_dir, name))
               for name in os.listdir(out_dir)
               if name.endswith(".safetensors"))
    _say(log, f"      wrote    {out_dir}  ({size / 1e9:.2f} GB)")

    # Given back before the caller loads it again, so the merge and whatever
    # loads the result are never both resident. On a pod sized for one 8B model
    # that difference is the run.
    del model
    import gc

    gc.collect()
    return out_dir
