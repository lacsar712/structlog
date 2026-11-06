# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Tests for the emission-time runtime policy stack (`structlog.emission`).
"""

import contextlib
import contextvars
import json
import logging
import threading

from io import StringIO

import pytest

import structlog

from structlog import (
    DENY_NEW,
    FORCE_EMIT,
    FREEZE_EVENT,
    HONOR_CFG,
    ONE_MORE,
    SKIP_CHAIN,
    BoundLogger,
    EventFrozenError,
    LogEmissionDenied,
    PrintLogger,
    WriteLogger,
    emission_scope,
    get_effective_policy,
    mute_below,
    pop_emission_policy,
    push_emission_policy,
)
from structlog.processors import JSONRenderer
from structlog.stdlib import (
    BoundLogger as StdlibBoundLogger,
)
from structlog.stdlib import (
    ProcessorFormatter,
)
from structlog.testing import CapturingLogger


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="cl")
def _cl():
    return CapturingLogger()


@pytest.fixture(name="native_bl")
def _native_bl(cl):
    return structlog.make_filtering_bound_logger(logging.INFO)(cl, [], {})


def _wrap(logger, processors=()):
    return BoundLogger(logger, list(processors), {})


def _inject(key, value):
    def proc(_logger, _name, event_dict):
        event_dict[key] = value
        return event_dict

    return proc


def _mutate(key, value):
    def proc(_logger, _name, event_dict):
        event_dict[key] = value
        return event_dict

    return proc


def _delete(key):
    def proc(_logger, _name, event_dict):
        del event_dict[key]
        return event_dict

    return proc


@pytest.fixture(name="stdlib_setup")
def _stdlib_setup():
    logger = logging.getLogger("structlog.test.emission")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    sio = StringIO()
    handler = logging.StreamHandler(sio)
    handler.setFormatter(
        ProcessorFormatter(
            processors=[
                ProcessorFormatter.remove_processors_meta,
                JSONRenderer(),
            ]
        )
    )
    logger.addHandler(handler)

    return logger, sio, handler


# ---------------------------------------------------------------------------
# Empty stack / pinning
# ---------------------------------------------------------------------------


class TestEmptyStack:
    def test_empty_policy_is_all_honor(self):
        policy = get_effective_policy()

        assert HONOR_CFG == policy.emit_admit
        assert HONOR_CFG == policy.processor_gate
        assert HONOR_CFG == policy.level_gate
        assert 0 == policy.depth

    def test_native_pin(self, native_bl, cl):
        native_bl.debug("nope")
        native_bl.info("yep")

        assert 1 == len(cl.calls)

    def test_push_pop_roundtrip(self):
        token = push_emission_policy(emit_admit=DENY_NEW)
        assert 1 == get_effective_policy().depth
        pop_emission_policy(token)

        assert 0 == get_effective_policy().depth

    def test_invalid_gate_values(self):
        with pytest.raises(ValueError, match="Invalid emit_admit"):
            push_emission_policy(emit_admit="nope")
        with pytest.raises(ValueError, match="Invalid processor_gate"):
            push_emission_policy(processor_gate="nope")
        with pytest.raises(ValueError, match="Invalid level_gate"):
            push_emission_policy(level_gate="nope")

    def test_mute_below_accepts_names(self):
        assert mute_below("warning") == mute_below(logging.WARNING)
        with pytest.raises(ValueError, match="Unknown log level"):
            mute_below("nope")

    def test_no_private_bag_on_loggers(self, cl):
        """
        The stack lives in contextvars, never on logger instances.
        """
        bl = _wrap(cl)
        pl = PrintLogger(StringIO())

        assert {"_logger", "_processors", "_context"} == set(vars(bl))
        assert {"_file", "_lock"} == set(vars(pl))


# ---------------------------------------------------------------------------
# emit_admit -- deny_new
# ---------------------------------------------------------------------------


class TestDenyNew:
    def test_generic_bound_logger(self, cl):
        bl = _wrap(cl)

        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied) as exc_info,
        ):
            bl.info("nope")

        assert "deny_new" == exc_info.value.reason
        assert "info" == exc_info.value.method_name
        assert [] == cl.calls

    def test_filtering_bound_logger(self, native_bl, cl):
        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            native_bl.warning("nope")

        assert [] == cl.calls

    def test_stdlib_bound_logger(self, stdlib_setup):
        logger, sio, _ = stdlib_setup
        bl = StdlibBoundLogger(
            logger, [ProcessorFormatter.wrap_for_formatter], {}
        )

        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            bl.info("nope")

        assert "" == sio.getvalue()

    def test_direct_print_logger(self):
        sio = StringIO()
        pl = PrintLogger(sio)

        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            pl.info("nope")
        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            pl.msg("nope-too")

        assert "" == sio.getvalue()

    def test_direct_write_logger(self):
        sio = StringIO()
        wl = WriteLogger(sio)

        with (
            pytest.raises(LogEmissionDenied),
            emission_scope(emit_admit=DENY_NEW),
        ):
            wl.error("nope")

        assert "" == sio.getvalue()

    def test_other_context_unaffected(self, cl):
        """
        A context copied before pushing still follows the old API.
        """
        bl = _wrap(cl)
        untouched = contextvars.copy_context()

        with emission_scope(emit_admit=DENY_NEW):
            untouched.run(bl.info, "still goes out")
            with pytest.raises(LogEmissionDenied):
                bl.info("blocked")

        assert 1 == len(cl.calls)

    def test_inner_honor_does_not_reopen(self, cl):
        bl = _wrap(cl)

        with (
            emission_scope(emit_admit=DENY_NEW),
            emission_scope(),  # all-honor inner frame
            pytest.raises(LogEmissionDenied),
        ):
            bl.info("still blocked")

    def test_exception_method_denied(self, native_bl, cl):
        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            native_bl.exception("boom")

        assert [] == cl.calls

    def test_emission_in_flight_not_cancelled(self, cl):
        """
        A deny pushed while an entry is running the chain must not cancel
        that entry; only later emissions are denied.
        """
        state = {}

        def push_deny(_logger, _name, event_dict):
            if "token" not in state:
                state["token"] = push_emission_policy(emit_admit=DENY_NEW)
            return event_dict

        bl = _wrap(cl, [push_deny, JSONRenderer()])

        bl.info("first")  # admitted before the processor pushed the frame
        assert 1 == len(cl.calls)

        with pytest.raises(LogEmissionDenied):
            bl.info("second")

        pop_emission_policy(state["token"])
        bl.info("third")
        assert 2 == len(cl.calls)
        assert 0 == get_effective_policy().depth


# ---------------------------------------------------------------------------
# emit_admit -- one_more
# ---------------------------------------------------------------------------


class TestOneMore:
    def test_exactly_one_admitted(self, cl):
        bl = _wrap(cl)

        with emission_scope(emit_admit=ONE_MORE):
            bl.info("first")
            with pytest.raises(LogEmissionDenied) as exc_info:
                bl.info("second")
            with pytest.raises(LogEmissionDenied):
                bl.info("third")

        assert "one_more" == exc_info.value.reason
        assert 1 == len(cl.calls)

    def test_second_emission_rereads_stack_top(self, cl):
        """
        After the one admission, an inner deny must be visible to the next
        emission -- no snapshot cached at the first emission.
        """
        bl = _wrap(cl)

        with emission_scope(emit_admit=ONE_MORE):
            bl.info("first")
            with (
                emission_scope(emit_admit=DENY_NEW),
                pytest.raises(LogEmissionDenied) as exc_info,
            ):
                bl.info("second")

            assert "deny_new" == exc_info.value.reason

    def test_muted_emission_does_not_consume(self, native_bl, cl):
        with emission_scope(
            emit_admit=ONE_MORE, level_gate=mute_below(logging.WARNING)
        ):
            assert native_bl.info("muted") is None  # swallowed, not consumed
            native_bl.warning("the one")
            with pytest.raises(LogEmissionDenied):
                native_bl.error("no budget left")

        assert 1 == len(cl.calls)

    def test_direct_native_write(self):
        sio = StringIO()
        pl = PrintLogger(sio)

        with emission_scope(emit_admit=ONE_MORE):
            pl.warning("first")
            with pytest.raises(LogEmissionDenied):
                pl.warning("second")

        assert "first\n" == sio.getvalue()


# ---------------------------------------------------------------------------
# processor_gate -- skip_chain
# ---------------------------------------------------------------------------


class TestSkipChain:
    def test_intermediate_processors_skipped(self, cl):
        bl = _wrap(cl, [_inject("injected", 42), JSONRenderer()])

        bl.info("hello")
        assert "injected" in cl.calls[0].args[0]

        with emission_scope(processor_gate=SKIP_CHAIN):
            bl.info("hello")

        rendered = cl.calls[1].args[0]
        assert "injected" not in rendered
        assert json.loads(rendered) == {"event": "hello"}

    def test_does_not_touch_configuration(self, cl):
        processors = [_inject("injected", 42), JSONRenderer()]
        structlog.configure(processors=processors)
        bl = structlog.get_logger()

        with emission_scope(processor_gate=SKIP_CHAIN):
            bl.info("hello")

        assert processors == structlog.get_config()["processors"]
        structlog.reset_defaults()

    def test_stdlib_formatter_chain(self, stdlib_setup):
        logger, sio, _ = stdlib_setup
        chain = [
            _inject("from_structlog_chain", 1),
            ProcessorFormatter.wrap_for_formatter,
        ]
        bl = StdlibBoundLogger(logger, chain, {})

        bl.info("honor")
        assert "from_structlog_chain" in sio.getvalue()

        with emission_scope(processor_gate=SKIP_CHAIN):
            bl.info("skipped")

        lines = sio.getvalue().strip().splitlines()
        assert "from_structlog_chain" not in lines[-1]
        assert json.loads(lines[-1])["event"] == "skipped"


# ---------------------------------------------------------------------------
# processor_gate -- freeze_event
# ---------------------------------------------------------------------------


class TestFreezeEvent:
    def test_overwrite_rejected(self, cl):
        bl = _wrap(cl, [_mutate("event", "hacked"), JSONRenderer()])

        with (
            emission_scope(processor_gate=FREEZE_EVENT),
            pytest.raises(EventFrozenError),
        ):
            bl.info("original")

    def test_delete_rejected(self, cl):
        bl = _wrap(cl, [_delete("event"), JSONRenderer()])

        with (
            emission_scope(processor_gate=FREEZE_EVENT),
            pytest.raises(EventFrozenError),
        ):
            bl.info("original")

    def test_bare_pop_rejected(self):
        from structlog.emission import _FrozenEventDict

        ed = _FrozenEventDict({"event": "x"})

        with pytest.raises(EventFrozenError):
            ed.pop("event")

    def test_append_still_works(self, cl):
        bl = _wrap(cl, [_inject("added", 1), JSONRenderer()])

        with emission_scope(processor_gate=FREEZE_EVENT):
            bl.info("hello")

        rendered = json.loads(cl.calls[0].args[0])
        assert "hello" == rendered["event"]
        assert 1 == rendered["added"]

    def test_console_renderer_still_writes_once(self):
        """
        The default console renderer consumes keys via pop-with-default;
        that must keep working and render the event exactly once.
        """
        sio = StringIO()
        renderer = structlog.dev.ConsoleRenderer(colors=False)
        pl = PrintLogger(sio)
        bl = BoundLogger(
            pl,
            [structlog.processors.add_log_level, renderer],
            {},
        )

        with emission_scope(processor_gate=FREEZE_EVENT):
            bl.info("frozen hello", x=1)

        out = sio.getvalue()
        assert "frozen hello" in out
        assert 1 == out.count("frozen hello")
        assert "x=1" in out

    def test_exception_path_with_console_renderer(self):
        sio = StringIO()
        pl = PrintLogger(sio)
        bl = BoundLogger(
            pl,
            [
                structlog.dev.set_exc_info,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            {},
        )

        try:
            raise ValueError("boom")
        except ValueError:
            with emission_scope(processor_gate=FREEZE_EVENT):
                bl.exception("it broke")

        out = sio.getvalue()
        assert "it broke" in out
        assert "ValueError: boom" in out

    def test_processor_exception_not_swallowed(self, cl):
        def boom(_logger, _name, event_dict):
            raise ValueError("processor failure")

        bl = _wrap(cl, [boom, JSONRenderer()])

        with (
            emission_scope(processor_gate=FREEZE_EVENT),
            pytest.raises(ValueError, match="processor failure"),
        ):
            bl.info("hello")

    def test_write_failure_not_swallowed(self):
        class FailingLogger:
            def info(self, *args, **kwargs):
                raise OSError("disk on fire")

        bl = _wrap(FailingLogger(), [JSONRenderer()])

        with (
            emission_scope(processor_gate=FREEZE_EVENT),
            pytest.raises(OSError, match="disk on fire"),
        ):
            bl.info("hello")

    def test_formatter_meta_still_removed(self, stdlib_setup):
        logger, sio, _ = stdlib_setup
        bl = StdlibBoundLogger(
            logger, [ProcessorFormatter.wrap_for_formatter], {}
        )

        with emission_scope(processor_gate=FREEZE_EVENT):
            bl.info("hello", x=1)

        rendered = json.loads(sio.getvalue().strip())
        assert "hello" == rendered["event"]
        assert 1 == rendered["x"]
        assert "_record" not in rendered
        assert "_from_structlog" not in rendered

    def test_formatter_mutation_rejected(self, stdlib_setup):
        logger, _sio, handler = stdlib_setup
        handler.setFormatter(
            ProcessorFormatter(
                processors=[
                    _mutate("event", "hacked"),
                    JSONRenderer(),
                ]
            )
        )
        # A structlog-origin record as wrap_for_formatter produces one.
        record = logging.LogRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            {"event": "original"},
            (),
            None,
        )
        record._logger = logger  # type: ignore[attr-defined]
        record._name = "info"  # type: ignore[attr-defined]

        with (
            emission_scope(processor_gate=FREEZE_EVENT),
            pytest.raises(EventFrozenError),
        ):
            handler.formatter.format(record)


# ---------------------------------------------------------------------------
# level_gate -- mute_below
# ---------------------------------------------------------------------------


class TestMuteBelow:
    def test_native_filtering_logger(self, native_bl, cl):
        with emission_scope(level_gate=mute_below(logging.WARNING)):
            assert native_bl.info("swallowed") is None
            assert not native_bl.is_enabled_for(logging.INFO)
            assert logging.WARNING == native_bl.get_effective_level()
            native_bl.warning("out")

        assert [c.method_name for c in cl.calls] == ["warning"]

    def test_generic_logger(self, cl):
        bl = _wrap(cl)

        with emission_scope(level_gate=mute_below(logging.WARNING)):
            assert bl.info("swallowed") is None
            bl.warning("out")

        assert 1 == len(cl.calls)

    def test_stdlib_chain_not_run(self, stdlib_setup):
        logger, sio, _ = stdlib_setup
        calls = []

        def counting_proc(_logger, _name, event_dict):
            calls.append(_name)
            return event_dict

        bl = StdlibBoundLogger(
            logger,
            [counting_proc, ProcessorFormatter.wrap_for_formatter],
            {},
        )

        with emission_scope(level_gate=mute_below(logging.ERROR)):
            bl.info("swallowed")
            bl.error("out")

        assert ["error"] == calls
        assert 1 == len(sio.getvalue().strip().splitlines())

    def test_direct_print_logger(self):
        sio = StringIO()
        pl = PrintLogger(sio)

        with emission_scope(level_gate=mute_below(logging.WARNING)):
            assert pl.info("swallowed") is None
            pl.warning("out")

        assert "out\n" == sio.getvalue()

    def test_exception_swallowed_at_error_floor(self, native_bl, cl):
        with emission_scope(level_gate=mute_below(logging.CRITICAL)):
            native_bl.exception("boom")

        assert [] == cl.calls

    def test_inner_pop_restores_outer(self, native_bl, cl):
        with emission_scope(level_gate=mute_below(logging.WARNING)):
            assert native_bl.info("muted outer") is None
            with emission_scope(level_gate=FORCE_EMIT):
                native_bl.debug("forced inner")
            assert native_bl.debug("muted again") is None

        native_bl.info("restored")
        assert 2 == len(cl.calls)


# ---------------------------------------------------------------------------
# level_gate -- force_emit
# ---------------------------------------------------------------------------


class TestForceEmit:
    def test_native_filtering_logger(self, native_bl, cl):
        with emission_scope(level_gate=FORCE_EMIT):
            native_bl.debug("forced")
            assert native_bl.is_enabled_for(logging.DEBUG)
            assert 0 == native_bl.get_effective_level()

        assert 1 == len(cl.calls)
        native_bl.debug("filtered again")
        assert 1 == len(cl.calls)

    async def test_native_async_filtering_logger(self, cl):
        bl = structlog.make_filtering_bound_logger(logging.WARNING)(cl, [], {})

        with emission_scope(level_gate=FORCE_EMIT):
            await bl.adebug("forced async")

        assert 1 == len(cl.calls)

    def test_stdlib_forces_through_level(self, stdlib_setup):
        logger, sio, _ = stdlib_setup
        logger.setLevel(logging.WARNING)
        bl = StdlibBoundLogger(
            logger,
            [
                structlog.stdlib.render_to_log_kwargs,
            ],
            {},
        )

        assert bl.debug("normally dropped") is None
        assert "" == sio.getvalue()

        with emission_scope(level_gate=FORCE_EMIT):
            bl.debug("forced")

        assert 1 == len(sio.getvalue().strip().splitlines())
        assert "forced" in sio.getvalue()

    def test_deny_still_wins_over_force(self, native_bl, cl):
        with (
            emission_scope(emit_admit=DENY_NEW, level_gate=FORCE_EMIT),
            pytest.raises(LogEmissionDenied),
        ):
            native_bl.debug("forced but denied")

        assert [] == cl.calls


# ---------------------------------------------------------------------------
# ProcessorFormatter: foreign records
# ---------------------------------------------------------------------------


class TestFormatterForeignRecords:
    def test_foreign_denied(self, stdlib_setup):
        logger, sio, handler = stdlib_setup
        formatter = handler.formatter
        record = logging.LogRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "foreign",
            (),
            None,
        )

        with (
            emission_scope(emit_admit=DENY_NEW),
            pytest.raises(LogEmissionDenied),
        ):
            formatter.format(record)

        assert "" == sio.getvalue()

    def test_foreign_muted(self, stdlib_setup):
        logger, sio, handler = stdlib_setup
        formatter = handler.formatter
        record = logging.LogRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "foreign",
            (),
            None,
        )

        with (
            emission_scope(level_gate=mute_below(logging.WARNING)),
            pytest.raises(LogEmissionDenied) as exc_info,
        ):
            formatter.format(record)

        assert "mute_below" == exc_info.value.reason
        assert "" == sio.getvalue()

    def test_foreign_honor_writes(self, stdlib_setup):
        logger, sio, handler = stdlib_setup
        record = logging.LogRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "foreign",
            (),
            None,
        )

        handler.emit(record)

        assert "foreign" in sio.getvalue()


# ---------------------------------------------------------------------------
# Context isolation / concurrency
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_two_threads_different_gates(self):
        results = {}
        barrier = threading.Barrier(2)

        def worker(name, gate):
            ctx = contextvars.copy_context()

            def body():
                scope = (
                    emission_scope(emit_admit=DENY_NEW)
                    if gate == DENY_NEW
                    else contextlib.nullcontext()
                )
                with scope:
                    barrier.wait(timeout=5)
                    sio = StringIO()
                    try:
                        PrintLogger(sio).warning("x")
                        denied = False
                    except LogEmissionDenied:
                        denied = True
                    results[name] = (denied, sio.getvalue())

            ctx.run(body)

        t1 = threading.Thread(target=worker, args=("deny", DENY_NEW))
        t2 = threading.Thread(target=worker, args=("ok", HONOR_CFG))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["deny"] == (True, "")
        assert results["ok"] == (False, "x\n")

    def test_one_more_per_context(self):
        counts = {}

        def worker():
            ctx = contextvars.copy_context()

            def body():
                sio = StringIO()
                pl = PrintLogger(sio)
                with emission_scope(emit_admit=ONE_MORE):
                    pl.msg("a")
                    with contextlib.suppress(LogEmissionDenied):
                        pl.msg("b")
                counts[threading.get_ident()] = sio.getvalue()

            ctx.run(body)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert {"a\n"} == set(counts.values())
        assert 3 == len(counts)

    async def test_asyncio_tasks_isolated(self, cl):
        other = CapturingLogger()

        async def denied_task():
            with emission_scope(emit_admit=DENY_NEW):
                bl = structlog.wrap_logger(
                    cl, processors=[], wrapper_class=BoundLogger
                )
                with pytest.raises(LogEmissionDenied):
                    bl.info("no")

        async def allowed_task():
            bl = structlog.wrap_logger(
                other, processors=[], wrapper_class=BoundLogger
            )
            with emission_scope(emit_admit=ONE_MORE):
                bl.info("yes")
                with pytest.raises(LogEmissionDenied):
                    bl.info("no again")

        import asyncio

        await asyncio.gather(denied_task(), allowed_task())

        assert [] == cl.calls
        assert [c.method_name for c in other.calls] == ["info"]


# ---------------------------------------------------------------------------
# Scope mechanics
# ---------------------------------------------------------------------------


class TestScopeMechanics:
    def test_exception_pops_frame(self, cl):
        bl = _wrap(cl)

        with (
            pytest.raises(RuntimeError, match="boom"),
            emission_scope(emit_admit=DENY_NEW),
        ):
            raise RuntimeError("boom")

        assert 0 == get_effective_policy().depth
        bl.info("after")
        assert 1 == len(cl.calls)

    def test_scope_as_decorator(self, cl):
        bl = _wrap(cl)

        @emission_scope(emit_admit=DENY_NEW)
        def critical_section():
            bl.info("nope")

        with pytest.raises(LogEmissionDenied):
            critical_section()

        assert 0 == get_effective_policy().depth

    def test_merged_view_nesting(self):
        with emission_scope(processor_gate=SKIP_CHAIN):
            policy = get_effective_policy()
            assert SKIP_CHAIN == policy.processor_gate
            assert 1 == policy.depth

            with emission_scope(processor_gate=FREEZE_EVENT):
                assert FREEZE_EVENT == get_effective_policy().processor_gate

            assert SKIP_CHAIN == get_effective_policy().processor_gate

        assert HONOR_CFG == get_effective_policy().processor_gate

    def test_contextvars_key_binding_unaffected(self, cl):
        structlog.contextvars.clear_contextvars()
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                JSONRenderer(),
            ]
        )
        bl = structlog.get_logger()

        with emission_scope(emit_admit=DENY_NEW):
            structlog.contextvars.bind_contextvars(request_id="42")
            with pytest.raises(LogEmissionDenied):
                bl.info("blocked but binding still works")

        try:
            assert (
                "42" == structlog.contextvars.get_contextvars()["request_id"]
            )
        finally:
            structlog.contextvars.clear_contextvars()
            structlog.reset_defaults()
