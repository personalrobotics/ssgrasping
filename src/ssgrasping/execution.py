"""Passive skill execution: a skill advances only when its caller advances it.

A skill is a sequence of stages written as a generator. Wherever it must wait, for an
ssrobot ``Execution`` to finish, for the runtime clock to reach a time, or for a plan
being made on a worker (lookahead), it yields a ``Wait``. ``SkillExecution.advance()``
resumes it while its wait is satisfied: it reads the clock, execution states, and plans,
and never steps, updates, or polls the runtime itself. Planning runs synchronously inside
``advance()``, or on a worker for a lookahead; that is computation, not time.

``run_skill`` is the synchronous convenience that drives the clock: it steps a manual
runtime, or updates an external one, through ``RobotContext``'s own calls. There is no
skill-owned thread, clock, or event loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any, Protocol

from ssrobot import ClockMode, Execution, RobotContext, SsrobotError

from ssgrasping._skill import Body, Failed, Run, Wait
from ssgrasping.outcomes import SkillResult, SkillState


class SkillExecution:
    """One run of a staged skill, advanced by its caller."""

    def __init__(self, run: Run, body: Body) -> None:
        self._run = run
        self._body = body
        self._wait: Wait | None = None
        self._state = SkillState.PENDING
        self._result: SkillResult | None = None

    @property
    def skill(self) -> str:
        return self._run.skill

    @property
    def state(self) -> SkillState:
        return self._state

    @property
    def stage(self) -> str | None:
        """The stage in progress, or the last one recorded once done."""
        return self._run.stage

    @property
    def active(self) -> Execution | None:
        """The ssrobot execution the skill is waiting on, while it is unfinished."""
        if self._wait is None or self._wait.execution is None or self._wait.execution.done:
            return None
        return self._wait.execution

    @property
    def planning(self) -> tuple[Future[Any], ...]:
        """The plans being made on a worker that the skill waits on, and nothing else: no
        runtime time needs to pass for it to continue, so a driver may block on them
        instead of stepping. Empty otherwise."""
        if self.done or self._wait is None or self._wait.future is None:
            return ()
        return (self._wait.future,)

    @property
    def result(self) -> SkillResult | None:
        return self._result

    @property
    def done(self) -> bool:
        return self._result is not None

    def advance(self) -> SkillState:
        """Run the skill until it waits on something not yet satisfied, or ends."""
        if self.done:
            return self._state
        self._state = SkillState.RUNNING
        ctx = self._run.ctx
        while self._wait is None or self._wait.satisfied(ctx):
            try:
                self._wait = next(self._body)
            except StopIteration:
                self._finish(self._run.ok())
                break
            except Failed:
                self._finish(SkillState.FAILED)
                break
            except SsrobotError as e:
                # An expected operational refusal: the stage in progress fails with it.
                self._run.fail(f"{e.code}: {e.message}")
                self._finish(SkillState.FAILED)
                break
        return self._state

    def cancel(self, reason: str = "cancelled by request") -> SkillState:
        """Cancel the active command, stop the skill, and record the state it left.
        No cleanup: what the robot holds, and what the scene declares, stay as they are."""
        if self.done:
            return self._state
        active = self.active
        if active is not None:
            self._run.ctx.cancel(active)
        self._body.close()
        self._run.cancelled(reason)
        self._finish(SkillState.CANCELLED)
        return self._state

    def _finish(self, state: SkillState) -> None:
        self._run.close()
        self._state = state
        self._wait = None
        self._result = self._run.result(state)


def run_skill(
    ctx: RobotContext,
    execution: SkillExecution,
    *,
    max_ticks: int | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float = 0.001,
) -> SkillResult:
    """Drive ``execution`` to its end through the context's clock, and return its result.

    A manual runtime is stepped, at most ``max_ticks`` times; an external one is updated
    every ``poll_interval_s`` seconds for at most ``timeout_s``. A skill still running
    when its bound is reached is cancelled."""
    if ctx.info.clock_mode is ClockMode.MANUAL:
        if max_ticks is None:
            raise ValueError("run_skill on a manual clock needs max_ticks")
        if not run_together(ctx, [execution], max_ticks=max_ticks):
            execution.cancel(f"run_skill bound of {max_ticks} ticks reached")
        bound = f"{max_ticks} ticks"
    else:
        if timeout_s is None:
            raise ValueError("run_skill on an external clock needs timeout_s")
        give_up = time.monotonic() + timeout_s
        ctx.update()
        execution.advance()
        while not execution.done and time.monotonic() < give_up:
            time.sleep(poll_interval_s)
            ctx.update()
            execution.advance()
        bound = f"{timeout_s} s"
    if not execution.done:
        execution.cancel(f"run_skill bound of {bound} reached")
    assert execution.result is not None
    return execution.result


class Passive(Protocol):
    """Anything advanced by its caller: a ``SkillExecution``, or ``Candidates``."""

    @property
    def done(self) -> bool: ...

    @property
    def planning(self) -> tuple[Future[Any], ...]: ...

    def advance(self) -> object: ...


def run_together(
    ctx: RobotContext,
    items: Sequence[Passive],
    *,
    max_ticks: int,
    tick: Callable[[], None] = lambda: None,
) -> bool:
    """Advance every one of ``items`` until all are done, on a manual clock, stepping it at
    most ``max_ticks`` times and calling ``tick`` after each step. While any unfinished
    item waits on the runtime, the clock steps. When every unfinished one waits only on
    plans being made on workers, the driver blocks on those plans instead: no runtime
    time passes, as when a stage plans synchronously, so the runtime's timeline does not
    depend on how long planning takes. Whether all finished."""
    for item in items:
        item.advance()
    ticks = 0
    while not all(item.done for item in items):
        pending = [item for item in items if not item.done]
        planning = [item.planning for item in pending]
        if all(planning):
            wait([f for futures in planning for f in futures], return_when=FIRST_COMPLETED)
        elif ticks >= max_ticks:
            return False
        else:
            ctx.step()
            ticks += 1
            tick()
        for item in pending:
            item.advance()
    return True
