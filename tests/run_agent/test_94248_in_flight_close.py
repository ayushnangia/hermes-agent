"""#94248: close() must not release TLS FDs or SessionDB from a stranger thread.

A delegated Codex worker that hits its 600s deadline is still blocked in
OpenSSL read. The parent then calls ``AIAgent.close()`` 17-72ms later.
``client.close()`` plus ``SessionDB.close()`` from that thread is the
#29507 / #70773 family: kernel FD recycle into SQLite, then SIGSEGV in
pysqlite_connection_execute.

The owning run_conversation thread (or a deferred close after it idles)
is the only one allowed to hard-close those resources.
"""
from __future__ import annotations

import threading


class _RecordingClient:
    def __init__(self):
        self.close_calls = 0
        self.close_threads = []
        self.is_closed = False

    def close(self):
        self.close_calls += 1
        self.close_threads.append(threading.current_thread().ident)
        self.is_closed = True


class _RecordingDB:
    def __init__(self):
        self.close_calls = 0
        self.end_session_calls = 0

    def close(self):
        self.close_calls += 1

    def end_session(self, *_a, **_k):
        self.end_session_calls += 1


def _bare_agent(**attrs):
    """AIAgent with __init__ bypassed, carrying only what close() reads."""
    from unittest.mock import patch

    with patch("run_agent.AIAgent.__init__", return_value=None):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sid-94248"
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = _RecordingClient()
    agent._session_db = _RecordingDB()
    agent._owns_session_db = True
    agent._end_session_on_close = True
    agent._session_messages = []
    agent.quiet_mode = True
    agent.provider = "openai-codex"
    agent.base_url = "https://example.test/v1"
    agent.model = "gpt-5.6"
    agent._execution_thread_id = None
    agent._run_conversation_active = False
    agent._run_conversation_idle = threading.Event()
    agent._run_conversation_idle.set()
    agent._deferred_close_scheduled = False
    agent._deferred_close_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._client_lock = threading.RLock()
    agent.shutdown_memory_provider = lambda *a, **k: None
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def _mark_stranger_in_flight(agent):
    """Simulate a live run_conversation on another thread (SSL still blocked)."""
    agent._run_conversation_active = True
    agent._execution_thread_id = (threading.current_thread().ident or 0) + 1_000_003
    agent._run_conversation_idle = threading.Event()  # uncleared: still in-flight
    agent._model_request_active.set()
    return agent._run_conversation_idle


def _release_in_flight(agent, idle=None):
    agent._run_conversation_active = False
    agent._model_request_active.clear()
    if idle is None:
        idle = getattr(agent, "_run_conversation_idle", None)
    if idle is not None:
        idle.set()
    deferred = getattr(agent, "_deferred_close_thread", None)
    if deferred is not None:
        deferred.join(timeout=2.0)
    return deferred


def test_idle_close_still_hard_closes_owned_sqlite_and_client():
    agent = _bare_agent()
    client = agent.client
    db = agent._session_db

    agent.close()

    assert client.close_calls == 1
    assert db.close_calls == 1
    assert db.end_session_calls == 1


def test_stranger_close_during_ssl_read_does_not_release_fds():
    agent = _bare_agent()
    client = agent.client
    db = agent._session_db
    idle = _mark_stranger_in_flight(agent)

    try:
        agent.close()

        assert client.close_calls == 0
        assert db.close_calls == 0
        assert db.end_session_calls == 0
        assert agent._owns_session_db is True
        assert getattr(agent, "_deferred_close_scheduled", False) is True
    finally:
        _release_in_flight(agent, idle)


def test_owner_thread_close_during_turn_still_hard_closes():
    """close() from the run_conversation thread is the FD-owner path."""
    agent = _bare_agent()
    client = agent.client
    db = agent._session_db
    agent._run_conversation_active = True
    agent._execution_thread_id = threading.current_thread().ident
    agent._model_request_active.set()

    agent.close()

    assert client.close_calls == 1
    assert db.close_calls == 1


def test_deferred_close_hard_closes_after_worker_idles():
    agent = _bare_agent()
    client = agent.client
    db = agent._session_db
    idle = _mark_stranger_in_flight(agent)

    agent.close()
    assert db.close_calls == 0
    assert client.close_calls == 0

    agent._run_conversation_active = False
    agent._model_request_active.clear()
    idle.set()

    deferred = getattr(agent, "_deferred_close_thread", None)
    assert deferred is not None
    deferred.join(timeout=2.0)
    assert not deferred.is_alive()

    assert db.close_calls == 1
    # Shared client was retired (shutdown only) and dropped on the first
    # close(); FD release is GC / owner-thread, never a stranger client.close().
    assert client.close_calls == 0
    assert agent._owns_session_db is False


def test_model_request_flag_alone_defers_hard_close():
    """Codex SSL read sets _model_request_active even if a test skipped the wrapper flag."""
    agent = _bare_agent()
    db = agent._session_db
    agent._execution_thread_id = (threading.current_thread().ident or 0) + 7
    agent._model_request_active.set()

    agent.close()

    assert db.close_calls == 0
    agent._model_request_active.clear()
    deferred = getattr(agent, "_deferred_close_thread", None)
    if deferred is not None:
        deferred.join(timeout=2.0)
    assert db.close_calls == 1


def test_should_defer_false_when_idle():
    from run_agent import AIAgent

    agent = _bare_agent()
    assert agent._should_defer_hard_close() is False
    assert isinstance(agent, AIAgent)


def test_hard_close_fence_releases_after_exception():
    agent = _bare_agent()

    try:
        with agent._fence_hard_close():
            assert agent._run_conversation_active is True
            assert not agent._run_conversation_idle.is_set()
            raise RuntimeError("turn failed")
    except RuntimeError:
        pass

    assert agent._run_conversation_active is False
    assert agent._run_conversation_idle.is_set()


def test_deferred_close_is_scheduled_once_under_concurrent_callers(monkeypatch):
    import run_agent as run_agent_module

    agent = _bare_agent()
    release = threading.Event()
    agent._wait_until_safe_to_hard_close = release.wait

    # Make the old check-then-set race deterministic: without the production
    # lock every caller snapshots False before any caller can publish True.
    # With the lock, only the first caller waits here; later callers observe
    # the published True and return without starting another close thread.
    flag_state = {"value": False, "false_reads": 0}
    flag_lock = threading.Lock()
    all_false_reads = threading.Event()

    def read_scheduled(_agent):
        snapshot = flag_state["value"]
        if not snapshot:
            with flag_lock:
                flag_state["false_reads"] += 1
                if flag_state["false_reads"] == 8:
                    all_false_reads.set()
            all_false_reads.wait(timeout=1.0)
        return snapshot

    def write_scheduled(_agent, value):
        flag_state["value"] = value

    monkeypatch.setattr(
        type(agent),
        "_deferred_close_scheduled",
        property(read_scheduled, write_scheduled),
        raising=False,
    )

    real_thread = threading.Thread
    created_deferred = []

    def recording_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        if kwargs.get("name") == "agent-deferred-close":
            created_deferred.append(thread)
        return thread

    monkeypatch.setattr(run_agent_module.threading, "Thread", recording_thread)

    callers_ready = threading.Barrier(9)

    def schedule():
        callers_ready.wait()
        agent._schedule_deferred_close()

    callers = [real_thread(target=schedule) for _ in range(8)]
    for caller in callers:
        caller.start()
    callers_ready.wait()
    for caller in callers:
        caller.join(timeout=2.0)

    assert len(created_deferred) == 1
    release.set()
    created_deferred[0].join(timeout=2.0)
    assert not created_deferred[0].is_alive()
    assert agent._deferred_close_scheduled is False


def test_in_flight_close_does_not_close_codex_session():
    class _Codex:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    session = _Codex()
    agent = _bare_agent(_codex_session=session)
    idle = _mark_stranger_in_flight(agent)

    try:
        agent.close()

        assert session.closed == 0
        assert agent._codex_session is session
    finally:
        _release_in_flight(agent, idle)
