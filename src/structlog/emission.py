# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Emission-time runtime policy stack.

This module hosts a :class:`~contextvars.ContextVar`-backed stack of
*emission policies* that is consulted while a log entry is being emitted --
independently of:

- the permanent ``processors`` / ``wrapper_class`` / ``logger_factory`` /
  filtering level configured through :func:`structlog.configure` or carried
  by a bound logger,
- the key-value binding APIs in :mod:`structlog.contextvars`,
- the Twisted observer integration, and
- the development console rendering.

A frame carries three independent gates:

- ``emit_admit`` decides whether *new* emissions are admitted at all
  (:data:`DENY_NEW`, :data:`ONE_MORE`),
- ``processor_gate`` decides how the configured processor chain is run
  (:data:`SKIP_CHAIN`, :data:`FREEZE_EVENT`),
- ``level_gate`` decides how the effective level filtering is changed for
  the frame (:class:`MuteBelow`, :data:`FORCE_EMIT`).

Frames are pushed/popped with :func:`push_emission_policy` /
:func:`pop_emission_policy` (or the :class:`emission_scope` context manager /
decorator) and are scoped to the current cooperative context: nested calls,
``asyncio`` tasks and threads spawned via :func:`contextvars.copy_context`
observe the stack *as it is at emission time*, never a snapshot taken when a
parent scope was entered.

Every gate defaults to :data:`HONOR_CFG`, which means that with an empty
stack the emission behavior is identical to the unpatched library.
"""

from __future__ import annotations

import contextlib
import contextvars

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, final

from ._log_levels import NAME_TO_LEVEL, NOTSET, map_method_name


__all__ = [
    "DENY_NEW",
    "FORCE_EMIT",
    "FREEZE_EVENT",
    "HONOR_CFG",
    "ONE_MORE",
    "SKIP_CHAIN",
    "EffectiveEmissionPolicy",
    "EmissionPolicyToken",
    "EventFrozenError",
    "LogEmissionDenied",
    "MuteBelow",
    "emission_scope",
    "get_effective_policy",
    "mute_below",
    "pop_emission_policy",
    "push_emission_policy",
]


# ---------------------------------------------------------------------------
# Gate values
# ---------------------------------------------------------------------------

HONOR_CFG = "honor_cfg"
"""
Behave exactly as permanently configured (the default for every gate).
"""

DENY_NEW = "deny_new"
"""
Reject all *new* log emissions started while the frame is active.

Entries that have already entered the processor chain or the final write when
the frame is pushed are not affected.
"""

ONE_MORE = "one_more"
"""
Admit exactly one more emission; reject everything that follows it.
"""

SKIP_CHAIN = "skip_chain"
"""
Skip the intermediate processors and run only the final rendering processor.
"""

FREEZE_EVENT = "freeze_event"
"""
Forbid processors from deleting or overwriting event keys present at method
entry; appending new keys remains allowed.
"""

FORCE_EMIT = "force_emit"
"""
Emit entries even at levels that the configured filtering would drop.
"""


@final
class MuteBelow:
    """
    Level gate value: tighten the effective minimum level for the frame.

    Emissions below *level* are swallowed and never reach the final write;
    the permanent configuration remains untouched.

    Use :func:`mute_below` to construct instances.
    """

    __slots__ = ("level",)

    level: int

    def __init__(self, level: int) -> None:
        self.level = level

    def __repr__(self) -> str:
        return f"mute_below({self.level})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MuteBelow) and other.level == self.level

    def __hash__(self) -> int:
        return hash((MuteBelow, self.level))


def mute_below(level: int | str) -> MuteBelow:
    """
    Create a :class:`MuteBelow` gate for *level*.

    *level* may be a numeric standard-library level or a level name.
    """
    if isinstance(level, str):
        try:
            level = NAME_TO_LEVEL[level.lower()]
        except KeyError:
            msg = f"Unknown log level: {level!r}"
            raise ValueError(msg) from None

    return MuteBelow(level)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LogEmissionDenied(RuntimeError):  # noqa: N818
    """
    Raised when an emission is rejected by an ``emit_admit`` frame
    (:data:`DENY_NEW` or an exhausted :data:`ONE_MORE`) or when a foreign
    standard-library record must be prevented from writing under a
    :class:`MuteBelow` frame inside a `~structlog.stdlib.ProcessorFormatter`.

    Attributes:
        reason: The gate that caused the rejection (``"deny_new"``,
            ``"one_more"`` or ``"mute_below"``).
        method_name: The wrapped-logger method the emission used.
    """

    def __init__(self, reason: str, method_name: str | None = None) -> None:
        self.reason = reason
        self.method_name = method_name
        super().__init__(reason, method_name)


class EventFrozenError(TypeError):
    """
    Raised when a processor tries to overwrite or explicitly delete an event
    key that was already present at the log method's entry while a
    :data:`FREEZE_EVENT` frame is active.
    """


# ---------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Frame:
    emit_admit: str
    processor_gate: str
    level_gate: Any  # HONOR_CFG / FORCE_EMIT / MuteBelow
    remaining: int = 1  # outstanding ONE_MORE admissions


# The var names must NOT start with the ``structlog_`` prefix used by
# structlog.contextvars' key-value binding API: get/clear_contextvars()
# enumerate ContextVars by that prefix and would otherwise expose or reset
# the policy stack as if it were a bound event value.
_STACK: contextvars.ContextVar[tuple[_Frame, ...]] = contextvars.ContextVar(
    "_emission_policy_stack", default=()
)

# Non-zero only while an already-admitted emission performs its *final* write
# onto the wrapped logger.  It marks tail writes (including a deferred
# ProcessorFormatter rendering) so that native write loggers do not evaluate
# the gates a second time for an entry that already passed them.
_WRITING: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_emission_writing", default=0
)


@final
class EmissionPolicyToken:
    """
    Opaque token returned by :func:`push_emission_policy` and consumed by
    :func:`pop_emission_policy`.
    """

    __slots__ = ("_previous",)

    def __init__(self, previous: tuple[_Frame, ...]) -> None:
        self._previous = previous


def _validate_emit_admit(value: str) -> None:
    if value not in (HONOR_CFG, DENY_NEW, ONE_MORE):
        msg = f"Invalid emit_admit value: {value!r}"
        raise ValueError(msg)


def _validate_processor_gate(value: str) -> None:
    if value not in (HONOR_CFG, SKIP_CHAIN, FREEZE_EVENT):
        msg = f"Invalid processor_gate value: {value!r}"
        raise ValueError(msg)


def _validate_level_gate(value: Any) -> None:
    if (
        value is not HONOR_CFG
        and value is not FORCE_EMIT
        and not isinstance(value, MuteBelow)
    ):
        msg = (
            "Invalid level_gate value: expected HONOR_CFG, FORCE_EMIT or "
            f"mute_below(level), got {value!r}"
        )
        raise ValueError(msg)


def push_emission_policy(
    *,
    emit_admit: str = HONOR_CFG,
    processor_gate: str = HONOR_CFG,
    level_gate: Any = HONOR_CFG,
) -> EmissionPolicyToken:
    """
    Push one emission-policy frame onto the stack of the current context.

    The frame stays in effect until :func:`pop_emission_policy` is called
    with the returned token.  Prefer :class:`emission_scope` for automatic,
    exception-safe popping.
    """
    _validate_emit_admit(emit_admit)
    _validate_processor_gate(processor_gate)
    _validate_level_gate(level_gate)

    stack = _STACK.get()
    token = EmissionPolicyToken(stack)
    _STACK.set(
        (
            *stack,
            _Frame(
                emit_admit=emit_admit,
                processor_gate=processor_gate,
                level_gate=level_gate,
            ),
        )
    )

    return token


def pop_emission_policy(token: EmissionPolicyToken) -> None:
    """
    Pop the frame identified by *token*, restoring the stack view the frame
    was pushed onto.
    """
    _STACK.set(token._previous)


class emission_scope(contextlib.ContextDecorator):  # noqa: N801
    """
    Context manager and decorator that pushes an emission-policy frame for
    its duration and pops it again on exit -- including when the body raises.

    Example::

        with emission_scope(emit_admit=DENY_NEW):
            log.info("not emitted")  # raises LogEmissionDenied
    """

    def __init__(
        self,
        *,
        emit_admit: str = HONOR_CFG,
        processor_gate: str = HONOR_CFG,
        level_gate: Any = HONOR_CFG,
    ) -> None:
        self._emit_admit = emit_admit
        self._processor_gate = processor_gate
        self._level_gate = level_gate
        self._token: EmissionPolicyToken | None = None

    def __enter__(self) -> EmissionPolicyToken:
        self._token = push_emission_policy(
            emit_admit=self._emit_admit,
            processor_gate=self._processor_gate,
            level_gate=self._level_gate,
        )

        return self._token

    def __exit__(self, *exc: object) -> None:
        pop_emission_policy(self._token)  # type: ignore[arg-type]
        self._token = None


class EffectiveEmissionPolicy(NamedTuple):
    """
    The merged view of the active stack.

    Attributes:
        emit_admit: Effective admission policy.
        processor_gate: Effective processor policy.
        level_gate: Effective level policy (:data:`HONOR_CFG`,
            :data:`FORCE_EMIT` or :class:`MuteBelow`).
        depth: Number of frames currently on the stack.
    """

    emit_admit: str
    processor_gate: str
    level_gate: Any
    depth: int


def _effective_gate(attribute: str) -> Any:
    stack = _STACK.get()
    for frame in reversed(stack):
        value = getattr(frame, attribute)
        if value is not HONOR_CFG:
            return value

    return HONOR_CFG


def current_emit_admit() -> str:
    """
    Return the effective admission policy (innermost non-honor frame wins).
    """
    return _effective_gate("emit_admit")


def current_processor_gate() -> str:
    """
    Return the effective processor policy (innermost non-honor frame wins).
    """
    return _effective_gate("processor_gate")


def frame_level_gate() -> Any:
    """
    Return the effective level policy (innermost non-honor frame wins).
    """
    return _effective_gate("level_gate")


def get_effective_policy() -> EffectiveEmissionPolicy:
    """
    Return the merged view of the current stack.

    With an empty stack every gate is :data:`HONOR_CFG` and ``depth`` is 0.
    """
    stack = _STACK.get()

    return EffectiveEmissionPolicy(
        emit_admit=current_emit_admit(),
        processor_gate=current_processor_gate(),
        level_gate=frame_level_gate(),
        depth=len(stack),
    )


# ---------------------------------------------------------------------------
# Gate hooks used by the emission exits
# ---------------------------------------------------------------------------


def admit_emission(method_name: str) -> None:
    """
    Evaluate the ``emit_admit`` axis for a *new* emission.

    Raises:
        LogEmissionDenied: if an active frame rejects the emission.
    """
    stack = _STACK.get()
    if not stack:
        return

    for index in range(len(stack) - 1, -1, -1):
        frame = stack[index]
        if frame.emit_admit == DENY_NEW:
            raise LogEmissionDenied(DENY_NEW, method_name)
        if frame.emit_admit == ONE_MORE:
            if frame.remaining > 0:
                # Spend the allowance inside *this* context only: replace the
                # frame with a consumed copy, never mutate a shared object.
                consumed = _Frame(
                    emit_admit=frame.emit_admit,
                    processor_gate=frame.processor_gate,
                    level_gate=frame.level_gate,
                    remaining=frame.remaining - 1,
                )
                _STACK.set((*stack[:index], consumed, *stack[index + 1 :]))

                return

            raise LogEmissionDenied(ONE_MORE, method_name)
        # HONOR_CFG: continue scanning outwards.


def native_level_allowed(level: int, configured_min: int) -> bool:
    """
    Merge the level axis with a statically filtered native logger's
    *configured_min* for an emission at *level*.
    """
    gate = frame_level_gate()
    if gate is HONOR_CFG:
        return level >= configured_min
    if gate is FORCE_EMIT:
        return True

    return level >= max(configured_min, gate.level)


def effective_level(configured_min: int) -> int:
    """
    Merge the level axis into an effective minimum level.
    """
    gate = frame_level_gate()
    if gate is FORCE_EMIT:
        return NOTSET
    if isinstance(gate, MuteBelow):
        return max(configured_min, gate.level)

    return configured_min


def wrapper_configured_floor(logger: Any) -> int:
    """
    Look up the permanent level floor of a wrapped logger.

    Only standard-library loggers expose meaningful level filtering here;
    everything else reports `structlog.NOTSET`.  The check deliberately does
    not duck-type arbitrary wrapped loggers, since calling
    ``getEffectiveLevel`` on them may itself be an observable action.
    """
    import logging

    if isinstance(logger, logging.Logger) and not logger.disabled:
        return logger.getEffectiveLevel()

    return NOTSET


def dispatch_wrapped(
    logger: Any,
    method_name: str,
    args: Sequence[Any],
    kw: Mapping[str, Any],
) -> Any:
    """
    Call *method_name* on *logger* with the processor-chain result.

    Under :data:`FORCE_EMIT` a standard-library logger that would filter the
    level gets the entry delivered through its private ``_log`` entry point
    instead, so the record reaches the handlers without touching the
    logger's permanent level.  Everything else is dispatched exactly as
    before.
    """
    if frame_level_gate() is FORCE_EMIT:
        level = NAME_TO_LEVEL.get(map_method_name(method_name))
        if level is not None and _is_filtering_stdlib_logger(logger, level):
            return _force_stdlib_logger(logger, level, args, kw)

    return getattr(logger, method_name)(*args, **kw)


def _is_filtering_stdlib_logger(logger: Any, level: int) -> bool:
    import logging

    return (
        isinstance(logger, logging.Logger)
        and not logger.disabled
        and not logger.isEnabledFor(level)
    )


def _force_stdlib_logger(
    logger: Any,
    level: int,
    args: Sequence[Any],
    kw: Mapping[str, Any],
) -> Any:
    # render_to_log_kwargs returns its result as a keyword mapping.
    call_args = tuple(args)
    call_kw = dict(kw)
    if not call_args and "msg" in call_kw:
        call_args = (call_kw.pop("msg"),)

    return logger._log(level, call_args[0], call_args[1:], **call_kw)


@contextlib.contextmanager
def writing() -> Iterator[None]:
    """
    Mark the final write of an admitted emission.

    Tail writes performed while this marker is active are not re-evaluated
    against the gates (they belong to an emission that already passed); an
    entry that started writing before a frame was pushed therefore can never
    be cancelled or rewritten by that frame.
    """
    token = _WRITING.set(_WRITING.get() + 1)
    try:
        yield
    finally:
        _WRITING.reset(token)


def guarded_native_write(logger: Any, method_name: str, message: Any) -> Any:
    """
    Apply the gates to a direct write on a native write logger
    (`structlog.PrintLogger`, `structlog.WriteLogger`, `structlog.BytesLogger`).

    Writes performed as the tail of an admitted bound-logger emission are
    passed through unchanged via the logger's raw write primitive.
    """
    if _WRITING.get():
        return logger._structlog_raw_write(message)

    level = NAME_TO_LEVEL.get(map_method_name(method_name))
    if level is not None:
        gate = frame_level_gate()
        if isinstance(gate, MuteBelow) and level < gate.level:
            return None
        # HONOR_CFG and FORCE_EMIT both mean "write" for a write logger that
        # has no level filtering of its own.

    admit_emission(method_name)

    return logger._structlog_raw_write(message)


def admit_foreign_record(levelno: int, method_name: str) -> None:
    """
    Apply admission and level gates to a *foreign* standard-library record
    that enters structlog through a `~structlog.stdlib.ProcessorFormatter`.

    Such a record can neither be silently dropped nor rewritten once it is
    being formatted, so a frame that requires suppression raises
    `LogEmissionDenied`; standard-library logging reports it through the
    normal handler error handling instead of writing the entry.
    """
    admit_emission(method_name)

    gate = frame_level_gate()
    if isinstance(gate, MuteBelow) and levelno < gate.level:
        raise LogEmissionDenied("mute_below", method_name)


# ---------------------------------------------------------------------------
# Frozen event dict
# ---------------------------------------------------------------------------


_MISSING = object()


class _FrozenEventDict(dict):  # type: ignore[type-arg]
    """
    Mapping that protects the keys captured at log-method entry.

    - Overwriting a live frozen key raises `EventFrozenError`.
    - ``del`` / bare ``pop`` / ``popitem`` / ``clear`` on live frozen keys
      raise `EventFrozenError`.
    - ``pop(key, default)`` of a live frozen key is *rejected* rather than
      executed: the original value is returned and the key becomes hidden
      from further mapping views, but it is never actually removed.  This
      keeps consuming renderers (which conventionally pop with a default)
      working without letting them destroy entry data.
    - Keys appended after entry can be modified and removed freely.
    """

    __slots__ = ("_frozen", "_hidden")

    def __init__(
        self,
        data: Any = (),
        *,
        frozen: set[str] | None = None,
    ) -> None:
        super().__init__(data)
        self._frozen = set(super().keys()) if frozen is None else set(frozen)
        self._hidden: set[Any] = set()

    def _live(self, key: Any) -> bool:
        return key in self._frozen and key not in self._hidden

    def _fail(self, key: Any) -> None:
        msg = (
            f"event key {key!r} is frozen for the current emission frame; "
            "processors may only append new keys"
        )
        raise EventFrozenError(msg)

    def __getitem__(self, key: Any) -> Any:
        if key in self._hidden:
            raise KeyError(key)

        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._live(key):
            self._fail(key)

        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        if self._live(key):
            self._fail(key)

        super().__delitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._hidden:
            return default

        return super().get(key, default)

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if self._live(key):
            if default is _MISSING:
                self._fail(key)
            # Reject the deletion while behaving like a successful consume.
            self._hidden.add(key)

            return super().__getitem__(key)

        if default is _MISSING:
            return super().pop(key)

        return super().pop(key, default)

    def popitem(self) -> tuple[Any, Any]:
        for key in super().__iter__():
            if self._live(key):
                self._fail(key)

        return super().popitem()

    def clear(self) -> None:
        for key in super().__iter__():
            if self._live(key):
                self._fail(key)

        super().clear()

    def update(self, *args: Any, **kwargs: Any) -> None:
        candidate: dict[Any, Any] = {}
        if args:
            data = args[0]
            if hasattr(data, "keys"):
                candidate = {key: data[key] for key in data}
            else:
                candidate = dict(data)

        candidate.update(kwargs)

        for key in candidate:
            if self._live(key):
                self._fail(key)

        super().update(candidate)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self._hidden:
            super().__setitem__(key, default)
            self._hidden.discard(key)

            return default

        if super().__contains__(key):
            return super().__getitem__(key)

        super().__setitem__(key, default)

        return default

    def __contains__(self, key: object) -> bool:
        return key not in self._hidden and super().__contains__(key)

    def __iter__(self) -> Iterator[Any]:
        for key in super().__iter__():
            if key not in self._hidden:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _visible(self) -> dict[Any, Any]:
        return {
            key: super().__getitem__(key)
            for key in super().__iter__()
            if key not in self._hidden
        }

    def keys(self) -> Any:
        return self._visible().keys()

    def items(self) -> Any:
        return self._visible().items()

    def values(self) -> Any:
        return self._visible().values()

    def copy(self) -> dict[Any, Any]:
        return self._visible()


# ---------------------------------------------------------------------------
# Gated processor chain execution
# ---------------------------------------------------------------------------


def select_processors(processors: Iterable[Any], gate: str) -> tuple[Any, ...]:
    """
    Return the processors to run under *gate* without touching the
    permanently configured chain.
    """
    chain = tuple(processors)
    if gate is SKIP_CHAIN and chain:
        return chain[-1:]

    return chain


def run_gated_processors(
    processors: Iterable[Any],
    logger: Any,
    method_name: str,
    event_dict: Any,
    *,
    frozen_keys: set[str] | None = None,
    chain_selector: Callable[[Iterable[Any], str], tuple[Any, ...]]
    | None = None,
) -> Any:
    """
    Run *processors* under the effective processor gate.

    *frozen_keys* overrides the set of keys considered frozen (used by the
    standard-library formatter, which appends its own metadata before
    running the chain).

    *chain_selector* overrides the skip-chain slicing (used by the
    standard-library formatter to keep metadata removal with the renderer).
    """
    gate = current_processor_gate()

    if gate is FREEZE_EVENT and not isinstance(event_dict, _FrozenEventDict):
        event_dict = _FrozenEventDict(event_dict, frozen=frozen_keys)

    selector = chain_selector or select_processors
    for processor in selector(processors, gate):
        event_dict = processor(logger, method_name, event_dict)

    return event_dict
