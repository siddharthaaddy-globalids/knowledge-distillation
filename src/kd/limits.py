"""
Ceilings a run may never cross: wall clock, optimizer steps, and money.

Two chances to catch an overrun, and the first one is the one that matters:

  BEFORE the run   the smoke stage measures seconds-per-step, which turns
                   max_steps into a projected duration and a projected cost. A
                   run that cannot finish inside its limits is refused here,
                   before anything expensive starts.

  DURING the run   a callback checks every step. A breach stops the process
                   immediately.

The in-flight check is a hard stop, deliberately: it guarantees the ceiling is
never crossed, at the price of ending with whatever the last periodic checkpoint
holds and no evaluation. That trade is why the pre-flight projection exists - the
common case should be "this run was never started", not "this run was killed at
73%". Tighten training.save_steps when limits are tight.

Cost is only meaningful when something is being rented. On a local machine
price_per_hour is None, so the money cap is inert and the time and step caps do
all the work.
"""

import time

from transformers import TrainerCallback


class LimitExceeded(RuntimeError):
    """A ceiling was crossed. Carries enough detail to explain the stop afterwards."""

    def __init__(self, limit, value, ceiling, unit):
        self.limit = limit
        self.value = value
        self.ceiling = ceiling
        self.unit = unit
        super().__init__(
            f"{limit} exceeded: {value:.2f}{unit} of a permitted {ceiling:.2f}{unit}")


class Budget:
    """What a run is allowed to spend, and how much of it has gone.

    price_per_hour is None for a local run - nothing is being rented, so there is
    no cost to cap. The RunPod launcher sets it to the rate it actually agreed to
    pay, which is the only number that makes a spend figure honest.
    """

    def __init__(self, config, price_per_hour=None, started=None):
        limits = config.get("limits") or {}
        self.max_runtime_minutes = limits.get("max_runtime_minutes")
        self.max_cost_usd = limits.get("max_cost_usd")
        self.confirm_above_usd = limits.get("confirm_above_usd")

        # The step ceiling is the tighter of the two: an explicit limits.max_steps
        # and whatever the training section asks for. A limit that sits above the
        # requested step count can never bind, so it is not worth reporting as one.
        requested = int(config.get("training", {}).get("max_steps") or 0)
        ceiling = limits.get("max_steps")
        self.max_steps = min(requested, int(ceiling)) if ceiling else requested

        self.price_per_hour = price_per_hour
        self.started = started or time.time()

    # --- where we are ------------------------------------------------------ #
    def elapsed_minutes(self):
        return (time.time() - self.started) / 60.0

    def spend_usd(self):
        """Money spent so far. Zero when nothing is rented."""
        if not self.price_per_hour:
            return 0.0
        return self.price_per_hour * (self.elapsed_minutes() / 60.0)

    def breach(self):
        """The limit that has been crossed, or None. Checked, not raised."""
        if self.max_runtime_minutes:
            elapsed = self.elapsed_minutes()
            if elapsed > self.max_runtime_minutes:
                return LimitExceeded("limits.max_runtime_minutes", elapsed,
                                     float(self.max_runtime_minutes), " min")
        if self.max_cost_usd and self.price_per_hour:
            spend = self.spend_usd()
            if spend > self.max_cost_usd:
                return LimitExceeded("limits.max_cost_usd", spend,
                                     float(self.max_cost_usd), " USD")
        return None

    def enforce(self):
        """Raise if a ceiling has been crossed."""
        breached = self.breach()
        if breached:
            raise breached

    # --- what is about to happen ------------------------------------------- #
    def project(self, seconds_per_step, steps=None):
        """Projected cost of a run at this measured pace.

        Returns a dict rather than a bare number because the caller wants to print
        all of it: the projection is only persuasive if the reader can see the
        measurement it came from.
        """
        steps = int(steps or self.max_steps or 0)
        minutes = (seconds_per_step * steps) / 60.0
        elapsed = self.elapsed_minutes()
        return {
            "seconds_per_step": seconds_per_step,
            "steps": steps,
            "train_minutes": minutes,
            # Whatever has already gone counts against the same ceilings.
            "total_minutes": elapsed + minutes,
            "cost_usd": (self.price_per_hour * ((elapsed + minutes) / 60.0)
                         if self.price_per_hour else None),
        }

    def refuse_if_impossible(self, projection):
        """Return a refusal message if this run cannot finish inside its limits.

        Refusing up front is the whole point of measuring the pace first: it is the
        difference between not spending the budget and spending most of it for an
        adapter you did not want.
        """
        if (self.max_runtime_minutes
                and projection["total_minutes"] > self.max_runtime_minutes):
            return (
                f"This run would take about {projection['total_minutes']:.0f} min, "
                f"over the {self.max_runtime_minutes} min limits.max_runtime_minutes.\n"
                f"  measured {projection['seconds_per_step']:.2f} s/step over "
                f"{projection['steps']} steps\n"
                f"  raise the ceiling:  --set limits.max_runtime_minutes="
                f"{int(projection['total_minutes'] * 1.2) + 1}\n"
                f"  or shorten the run: --set training.max_steps="
                f"{self._steps_that_fit(projection)}")

        cost = projection.get("cost_usd")
        if self.max_cost_usd and cost and cost > self.max_cost_usd:
            return (
                f"This run would cost about ${cost:.2f}, over the "
                f"${self.max_cost_usd:.2f} limits.max_cost_usd.\n"
                f"  measured {projection['seconds_per_step']:.2f} s/step over "
                f"{projection['steps']} steps at ${self.price_per_hour:.2f}/hr\n"
                f"  raise the ceiling:  --set limits.max_cost_usd={cost * 1.2:.2f}\n"
                f"  or shorten the run: --set training.max_steps="
                f"{self._steps_that_fit(projection)}")
        return None

    def _steps_that_fit(self, projection):
        """The largest step count that would fit inside every ceiling."""
        per_step = projection["seconds_per_step"] or 1.0
        elapsed = self.elapsed_minutes()
        candidates = []
        if self.max_runtime_minutes:
            candidates.append((self.max_runtime_minutes - elapsed) * 60.0 / per_step)
        if self.max_cost_usd and self.price_per_hour:
            budget_minutes = (self.max_cost_usd / self.price_per_hour) * 60.0 - elapsed
            candidates.append(budget_minutes * 60.0 / per_step)
        # A tenth off, so the suggestion is not itself borderline.
        return max(1, int(min(candidates) * 0.9)) if candidates else projection["steps"]

    def summary(self):
        """One line per active ceiling, for the startup banner."""
        lines = []
        if self.max_runtime_minutes:
            lines.append(f"{self.max_runtime_minutes} min wall clock")
        if self.max_steps:
            lines.append(f"{self.max_steps} steps")
        if self.max_cost_usd:
            lines.append(f"${self.max_cost_usd:.2f}"
                         + ("" if self.price_per_hour else " (inert: nothing rented)"))
        return ", ".join(lines) or "none"


class LimitCallback(TrainerCallback):
    """Enforces the budget during training, and records spend as the run proceeds.

    Raising from a callback propagates out of Trainer.train(), which is what makes
    this a hard stop rather than a request the training loop may decline.
    """

    def __init__(self, budget, run=None, tick_every=10):
        self.budget = budget
        self.run = run
        self.tick_every = tick_every

    def on_step_end(self, args, state, control, **kwargs):
        if self.budget.max_steps and state.global_step > self.budget.max_steps:
            raise LimitExceeded("limits.max_steps", float(state.global_step),
                                float(self.budget.max_steps), " steps")
        self.budget.enforce()

        if self.tick_every and state.global_step % self.tick_every == 0 and self.run:
            self.run.event("train", "budget", step=state.global_step,
                           elapsed_minutes=round(self.budget.elapsed_minutes(), 2),
                           spend_usd=round(self.budget.spend_usd(), 4))
