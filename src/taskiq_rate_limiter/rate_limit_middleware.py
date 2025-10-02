import asyncio
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from taskiq.abc.middleware import TaskiqMiddleware
from taskiq.message import TaskiqMessage

logger = getLogger("taskiq_rate_limiter.rate_limit_middleware")


@dataclass(slots=True)
class RateLimitMetrics:
    """Lightweight counters exposed by the limiter."""

    executed_in_window: int = 0
    queued_total: int = 0
    rejected_total: int = 0
    current_queue_depth: int = 0
    next_allowed_at: Optional[datetime] = None


@dataclass(slots=True)
class RateLimitConfig:
    """Configuration for the per-process rate limiter."""

    # Maximum starts allowed inside a single window.
    limit: int = 100
    # Duration of the rolling window in seconds.
    window_seconds: float = 60.0
    # Optional cap for pending callers waiting on permits (``None`` = unbounded).
    max_queue_size: Optional[int] = None
    # If ``True`` raise when the queue is full; otherwise callers retry later.
    reject_when_full: bool = False
    # Optional number of immediate starts before pacing activates.
    pacing_start_threshold: Optional[int] = None
    # The pacing strategy to use.
    # "adaptive": (Default) Prioritizes smoothing load by spacing tasks based on
    #             remaining time and slots. May not use all slots in a window.
    # "fixed":    Prioritizes throughput by using a fixed pace, ensuring all slots
    #             in a window can be used if tasks arrive in time.
    pacing_strategy: Literal["adaptive", "fixed"]  = "adaptive"
    # Upper bound (in milliseconds) for jitter applied to scheduled delays.
    jitter_ms: float = 0
    # Optional hook invoked with the next permitted wall-clock timestamp.
    on_next_allowed: Optional[Callable[[datetime], None]] = None

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be a positive integer")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        if self.max_queue_size is not None and self.max_queue_size < 0:
            raise ValueError("max_queue_size cannot be negative")
        if self.pacing_start_threshold is not None and self.pacing_start_threshold < 0:
            raise ValueError("pacing_start_threshold cannot be negative")
        if self.pacing_strategy not in ("adaptive", "fixed"):
            raise ValueError("pacing_strategy must be 'adaptive' or 'fixed'")
        if self.jitter_ms < 0:
            raise ValueError("jitter_ms cannot be negative")


class RateLimitQueueFullError(RuntimeError):
    """Raised when new arrivals are rejected due to a bounded queue."""


@dataclass(slots=True)
class _ScheduleDecision:
    delay: float
    permit: "_Permit"
    retry_delay: Optional[float] = None


class TaskStartRateLimiter:
    """Async rate limiter that evens out task start times within a window."""

    def __init__(
        self,
        config: RateLimitConfig,
        *,
        time_source: Callable[[], float] = time.monotonic,
        wall_time_source: Callable[[], float] = time.time,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_source: Optional[random.Random] = None,
    ) -> None:
        """Create a limiter bound to a configuration and timing primitives.

        Args:
            config: Static limiter parameters shared across workers in the process.
            time_source: Monotonic time provider used for all scheduling math.
            wall_time_source: Wall-clock provider used for instrumentation callbacks.
            sleep_func: Awaitable used to pause between permit checks.
            random_source: Optional RNG used to apply jitter;
            Defaults to ``random.Random``.
        """
        if config.limit <= 0:
            raise ValueError("limit must be greater than zero")
        if config.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")

        self._cfg = config
        self._time = time_source
        self._wall = wall_time_source
        self._sleep = sleep_func
        self._rand = random_source or random.Random()  # noqa: S311
        self._condition = asyncio.Condition()
        self._window_start: Optional[float] = None
        self._window_end: Optional[float] = None
        self._granted_in_window = 0
        self._executed_in_window = 0
        self._waiting_count = 0
        self._next_allowed_time: Optional[float] = None
        self._pacing_threshold = self._resolve_threshold(config)
        self._epsilon = 1e-9
        self.metrics = RateLimitMetrics()

    @staticmethod
    def _resolve_threshold(config: RateLimitConfig) -> int:
        if config.pacing_start_threshold is not None:
            return max(0, min(config.limit, config.pacing_start_threshold))
        if config.limit <= 2:
            return 0
        return max(1, config.limit // 2)

    async def throttle(self, task_name: Optional[str] = None) -> None:
        """Block the caller until it can start executing inside the current window.

        Args:
            task_name: Optional human-readable label used for log messages.

        Returns:
            None. The coroutine resumes only when execution is permitted.
        """
        label = task_name or "task"
        while True:
            async with self._condition:
                now = self._time()
                self._rotate_window(now)
                decision = self._compute_schedule(now, label)
                if decision.retry_delay is None:
                    delay = max(0.0, decision.delay)
                    break
                retry = max(0.0, decision.retry_delay)
            if retry > 0:
                await self._sleep(retry)
            else:
                await asyncio.sleep(0)
        try:
            if delay > 0:
                logger.debug("Rate limiting %s for %.3f seconds", label, delay)
            await self._sleep(delay)
        finally:
            await decision.permit.mark_started()

    def _rotate_window(self, now: float) -> None:
        """Initialise or roll the active window if ``now`` moved past its end."""
        window_len = self._cfg.window_seconds
        if self._window_start is None or self._window_end is None:
            self._window_start = now
            self._window_end = now + window_len
            self._granted_in_window = 0
            self._executed_in_window = 0
            self._next_allowed_time = now
            self.metrics.executed_in_window = 0
            self.metrics.current_queue_depth = self._waiting_count
            logger.info(
                "Rate limiter window initialised: limit=%d, window=%.3fs",
                self._cfg.limit,
                window_len,
            )
            return

        if now < self._window_end:
            return

        # Handle large jumps by skipping whole windows in one go.
        windows_passed = max(1, math.floor((now - self._window_start) / window_len))
        self._window_start = self._window_start + windows_passed * window_len
        if now >= self._window_start + window_len:
            self._window_start = now
        self._window_end = self._window_start + window_len
        self._granted_in_window = 0
        self._executed_in_window = 0
        self._next_allowed_time = self._window_start
        self.metrics.executed_in_window = 0
        self.metrics.current_queue_depth = self._waiting_count
        logger.info(
            "Rate limiter window rotated: start=%.3f, end=%.3f",
            self._window_start,
            self._window_end,
        )

    def _compute_schedule(self, now: float, label: str) -> _ScheduleDecision:  # noqa: C901, PLR0912, PLR0915
        """Compute pacing delay and updated accounting for one permit request.

        Args:
            now: Current monotonic timestamp.
            label: Human-readable identifier used for logging.

        Returns:
            A scheduling decision describing when to run and any retry delay.
        """
        assert self._window_start is not None and self._window_end is not None  # noqa: S101

        remaining_slots = self._cfg.limit - self._granted_in_window
        if remaining_slots <= 0:
            wait_for = max(0.0, self._window_end - now)
            logger.info(
                "Rate limit exhausted for %s: deferring %.3f seconds until next window",
                label,
                wait_for,
            )
            if wait_for <= self._epsilon:
                wait_for = self._cfg.window_seconds
            self._emit_next_allowed(now + wait_for, now)
            return _ScheduleDecision(
                delay=0.0,
                permit=_Permit(self, False),
                retry_delay=wait_for,
            )

        remaining_seconds = max(0.0, self._window_end - now)
        if remaining_seconds <= self._epsilon:
            self._emit_next_allowed(now, now)
            return _ScheduleDecision(
                delay=0.0,
                permit=_Permit(self, False),
                retry_delay=self._cfg.window_seconds,
            )

        # --- PACING STRATEGY LOGIC ---
        pacing_active = self._granted_in_window >= self._pacing_threshold
        target_time = now
        spacing = 0.0  # Used by adaptive strategy for updating next_allowed_time

        if pacing_active:
            if self._cfg.pacing_strategy == "fixed":
                # "Fixed" strategy: Prioritizes throughput.
                # Calculates a fixed pace based on the total window size to ensure
                # all slots can be used.
                fixed_pace = self._cfg.window_seconds / self._cfg.limit
                # The ideal start time is based on which permit number this is.
                ideal_scheduled_time = self._window_start + (
                    self._granted_in_window * fixed_pace
                )
                # The task must start now or later, whichever is the ideal time.
                target_time = max(now, ideal_scheduled_time)
            else:
                # "Adaptive" strategy: Prioritizes smoothing.
                spacing = remaining_seconds / remaining_slots
                basis = self._next_allowed_time or now
                target_time = max(now + spacing, basis)

        if target_time >= self._window_end - self._epsilon:
            wait_for = max(0.0, self._window_end - now)
            self._emit_next_allowed(now + wait_for, now)
            return _ScheduleDecision(
                delay=0.0,
                permit=_Permit(self, False),
                retry_delay=wait_for,
            )

        raw_delay = max(0.0, target_time - now)
        jitter = self._compute_jitter(raw_delay, remaining_seconds)
        actual_delay = min(
            raw_delay + jitter,
            max(0.0, self._window_end - now - self._epsilon),
        )
        was_queued = actual_delay > self._epsilon

        if (
            was_queued
            and self._cfg.max_queue_size is not None
            and self._waiting_count >= self._cfg.max_queue_size
        ):
            if self._cfg.reject_when_full:
                self.metrics.rejected_total += 1
                logger.warning("Rate limit queue full for %s", label)
                raise RateLimitQueueFullError("Rate limit queue is full")
            logger.info(
                "Rate limit queue saturated for %s: retrying after %.3f seconds",
                label,
                actual_delay,
            )
            self._emit_next_allowed(now + actual_delay, now)
            return _ScheduleDecision(
                delay=0.0,
                permit=_Permit(self, False),
                retry_delay=actual_delay,
            )

        scheduled_time = now + actual_delay

        if pacing_active:
            if self._cfg.pacing_strategy == "fixed":
                fixed_pace = self._cfg.window_seconds / self._cfg.limit
                # The next allowed time is the next fixed, ideal slot.
                # We use `_granted_in_window + 1`
                # because the current one hasn't been counted yet.
                self._next_allowed_time = self._window_start + (
                    (self._granted_in_window + 1) * fixed_pace
                )
            else:  # "adaptive"
                self._next_allowed_time = scheduled_time + spacing
        else:
            base = self._next_allowed_time or scheduled_time
            self._next_allowed_time = max(base, scheduled_time)

        self._granted_in_window += 1
        if was_queued:
            self._waiting_count += 1
            self.metrics.queued_total += 1
        self.metrics.current_queue_depth = self._waiting_count
        self._emit_next_allowed(scheduled_time, now)

        logger.info(
            "Rate limit decision for %s: delay=%.3fs, "
            "pacing=%s, remaining_slots=%d, queue_depth=%d",
            label,
            actual_delay,
            pacing_active,
            self._cfg.limit - self._granted_in_window,
            self._waiting_count,
        )

        return _ScheduleDecision(delay=actual_delay, permit=_Permit(self, was_queued))

    def _emit_next_allowed(self, when: float, base_now: float) -> None:
        """Record the next permissible wall-clock execution time.

        Args:
            when: Monotonic timestamp representing the next scheduled start.
            base_now: Monotonic timestamp representing the decision point.

        Returns:
            None.
        """
        wall_ts = self._wall() + max(0.0, when - base_now)
        ts = datetime.fromtimestamp(wall_ts, tz=timezone.utc)
        self.metrics.next_allowed_at = ts
        if self._cfg.on_next_allowed is not None:
            try:
                self._cfg.on_next_allowed(ts)
            except Exception:  # pragma: no cover - instrumentation only
                logger.exception("on_next_allowed callback failed")

    def _compute_jitter(self, delay: float, remaining_seconds: float) -> float:
        """Return randomised slack so simultaneous arrivals do not align perfectly.

        Args:
            delay: Nominal delay (seconds) before executing the caller.
            remaining_seconds: Seconds left before the window closes.

        Returns:
            A jitter value in seconds, clamped to the remaining window.
        """
        if self._cfg.jitter_ms <= 0:
            return 0.0
        jitter = self._rand.uniform(0.0, self._cfg.jitter_ms / 1000.0)
        return min(jitter, max(0.0, remaining_seconds - delay))

    async def _mark_started(self, from_queue: bool) -> None:
        """Update metrics and wake queued waiters when an execution begins.

        Args:
            from_queue: ``True`` when this permit was previously queued.

        Returns:
            None.
        """
        async with self._condition:
            now = self._time()
            self._rotate_window(now)
            if from_queue and self._waiting_count > 0:
                self._waiting_count -= 1
                self._condition.notify_all()
            self._executed_in_window += 1
            self.metrics.executed_in_window = self._executed_in_window
            self.metrics.current_queue_depth = self._waiting_count
            logger.info(
                "Task marked started: executed=%d, queue_depth=%d",
                self._executed_in_window,
                self._waiting_count,
            )


@dataclass(slots=True)
class _Permit:
    limiter: TaskStartRateLimiter
    was_queued: bool

    async def mark_started(self) -> None:
        await self.limiter._mark_started(self.was_queued)  # noqa: SLF001


class RateLimitMiddleware(TaskiqMiddleware):
    """
    Middleware to rate limit tasks based on labels.

    This middleware allows you to configure rate limiting on a per-task
    basis by adding labels to your tasks.

    Example:
    .. code-block:: python

        @broker.task(
            rate_limit_enabled=True,
            rate_limit_limit=100,
            rate_limit_window_seconds=60,
            rate_limit_pacing_strategy="fixed",
        )
        async def my_limited_task():
            ...

    """

    def __init__(
        self,
        default_rate_limit_enabled: bool = False,
        default_limit: int = 60,
        default_window_seconds: float = 60.0,
        default_max_queue_size: Optional[int] = None,
        default_reject_when_full: bool = False,
        default_pacing_start_threshold: Optional[int] = None,
        default_pacing_strategy: Literal["adaptive", "fixed"] = "adaptive",
        default_jitter_ms: float = 0,
    ) -> None:
        """
        Initialize the rate limit middleware.

        :param default_rate_limit_enabled: Whether to enable rate limiting by default.
        :param default_limit: Default maximum starts in a window.
        :param default_window_seconds: Default duration of the rolling window.
        :param default_max_queue_size: Default cap for pending callers.
        :param default_reject_when_full: Default behavior when queue is full.
        :param default_pacing_start_threshold: Default immediate starts before pacing.
        :param default_pacing_strategy: Default pacing algorithm('adaptive' or 'fixed').
        :param default_jitter_ms: Default upper bound for jitter.
        """
        super().__init__()
        self.default_rate_limit_enabled = default_rate_limit_enabled
        self.defaults = {
            "limit": default_limit,
            "window_seconds": default_window_seconds,
            "max_queue_size": default_max_queue_size,
            "reject_when_full": default_reject_when_full,
            "pacing_start_threshold": default_pacing_start_threshold,
            "pacing_strategy": default_pacing_strategy,
            "jitter_ms": default_jitter_ms,
        }
        self.limiters: Dict[str, TaskStartRateLimiter] = {}
        self._lock = asyncio.Lock()

    def _is_enabled(self, message: TaskiqMessage) -> bool:
        """Check if rate limiting is enabled for a given task message."""
        enabled = message.labels.get("rate_limit_enabled")
        if enabled is None:
            return self.default_rate_limit_enabled
        if isinstance(enabled, str):
            return enabled.lower() == "true"
        return bool(enabled)

    async def _get_limiter(self, message: TaskiqMessage) -> TaskStartRateLimiter:
        """Get or create a rate limiter for a given task."""
        task_name = message.task_name
        if task_name in self.limiters:
            return self.limiters[task_name]

        async with self._lock:
            # Double-checked locking pattern
            if task_name in self.limiters:
                return self.limiters[task_name]

            labels = message.labels

            def _get_label(key: str, type_cast: Callable[[Any], Any]) -> Any:
                label_key = f"rate_limit_{key}"
                if label_key in labels:
                    return type_cast(labels[label_key])
                return self.defaults[key]

            def _get_bool_label(key: str) -> bool:
                label_key = f"rate_limit_{key}"
                val = labels.get(label_key)
                if isinstance(val, str):
                    return val.lower() == "true"
                if val is None:
                    return bool(self.defaults[key])
                return bool(val)

            config = RateLimitConfig(
                limit=_get_label("limit", int),
                window_seconds=_get_label("window_seconds", float),
                max_queue_size=_get_label("max_queue_size", int),
                reject_when_full=_get_bool_label("reject_when_full"),
                pacing_start_threshold=_get_label("pacing_start_threshold", int),
                pacing_strategy=_get_label("pacing_strategy", str),
                jitter_ms=_get_label("jitter_ms", float),
            )

            limiter = TaskStartRateLimiter(config)
            self.limiters[task_name] = limiter
            logger.info(
                "Created new rate limiter for task '%s' with config: %s",
                task_name,
                config,
            )
            return limiter

    async def pre_execute(self, message: TaskiqMessage) -> TaskiqMessage:
        """Pause task execution until the limiter allows it to start."""
        if not self._is_enabled(message):
            return message

        limiter = await self._get_limiter(message)
        await limiter.throttle(message.task_name)
        return message
