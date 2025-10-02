# Taskiq Rate Limiter Middleware

An advanced, in-process, asynchronous rate-limiting middleware for the [Taskiq](https://github.com/taskiq-python/taskiq) framework.

This middleware offers advanced pacing strategies designed to either **smooth task execution** to prevent load spikes or **maximize task throughput** by ensuring a specific number of tasks are processed within a time window.

[Video](https://github.com/user-attachments/assets/12bb3937-1f40-4d50-bd5d-83c7b80c455c)

## Key Features

-   **Rolling Window Rate Limiting:** Restrict the number of tasks that can start within a configurable time window (e.g., 100 tasks per 60 seconds).
-   **Burst Control:** Allow a configurable number of tasks to execute immediately before pacing is enforced. (Default is half of the limit)
-   [**Configurable Pacing Strategies:**](#deep-dive-pacing-strategies)
    -   `adaptive` (default): Prioritizes **load smoothing** by dynamically spacing tasks based on the remaining time and capacity in the current window.
    -   `fixed`: Prioritizes **throughput maximization** by enforcing a fixed, predictable interval between tasks to guarantee the full limit is utilized over the window.
-   **Task Queuing:** Asynchronously queues tasks that exceed the rate limit by default, rather than rejecting them.
-   **Jitter:** Introduces a small amount of random jitter to task start times, helping to prevent the "thundering herd" problem in distributed systems.

## Installation


```sh
pip install taskiq-rate-limiter
# OR
uv add taskiq-rate-limiter
```

## Quick Start

Integrate the `RateLimitMiddleware` into your Taskiq broker and configure tasks for rate limiting using labels.

```python
from taskiq import InMemoryBroker
from rate_limit_middleware import RateLimitMiddleware

# 1. Initialize the middleware with global defaults
# Use the 'fixed' strategy to ensure a consistent 100 tasks per minute.
rate_limiter = RateLimitMiddleware(
    default_rate_limit_enabled=True,
    default_limit=100,
    default_window_seconds=60,
    default_pacing_strategy="fixed"
)

# 2. Add the middleware to your broker
broker = InMemoryBroker().with_middlewares(rate_limiter)

# 3. This task will inherit the global rate-limiting settings
@broker.task
async def my_limited_task(i: int):
    print(f"Executing task {i}")
    return i

# ---- OR: Per-Task Configuration -----

# Disable rate limiting by default
rate_limiter = RateLimitMiddleware(default_rate_limit_enabled=False)
broker = InMemoryBroker().with_middlewares(rate_limiter)

# And enable it specifically on tasks that need it
@broker.task(
    labels={
        "rate_limit_enabled": True,
        "rate_limit_limit": 100,
        "rate_limit_window_seconds": 60,
        "rate_limit_pacing_strategy": "fixed",
    }
)
async def my_limited_task(i: int):
    print(f"Executing task {i}")
    return i
```

## Configuration Options

Configuration is applied in two layers:
1.  **Global Defaults:** Set during `RateLimitMiddleware(...)` initialization.
2.  **Per-Task Overrides:** Set as key-value pairs in the `labels` dictionary of the `@broker.task(...)` decorator.

Below is a complete list of available options.

### 1. Core Limiting

These settings define the fundamental rate limit parameters.

*   **`rate_limit_enabled`** (bool): Enables or disables rate limiting for the task.
    *   *Default:* `False` (unless overridden in middleware init).
*   **`rate_limit_limit`** (int): The maximum number of tasks allowed to start within the window.
    *   *Default:* `100`
*   **`rate_limit_window_seconds`** (float): The duration of the rolling window in seconds.
    *   *Default:* `60.0`

```python
# Allow 500 tasks every 5 minutes (300 seconds)
@broker.task(labels={
    "rate_limit_enabled": True,
    "rate_limit_limit": 500,
    "rate_limit_window_seconds": 300,
})
async def bulk_process(): ...
```

### 2. Pacing & Burst Control

These options control the execution timing of tasks within the window.

*   **`rate_limit_pacing_strategy`** (str): Determines the pacing algorithm. Accepts `"adaptive"` (for load smoothing) or `"fixed"` (for throughput maximization).
    *   *Default:* `"adaptive"`
*   **`rate_limit_pacing_start_threshold`** (int): The number of tasks allowed to execute immediately (burst) at the start of a window before the pacing strategy is applied.
    *   *Default:* `limit // 2`. Set to `0` to pace every task. Set to `limit` to disable pacing (pure burst).

```python
# Example: Strict, even pacing for an SLA.
# Process exactly 1 task every second (60 per 60s) with no initial burst.
@broker.task(labels={
    "rate_limit_enabled": True,
    "rate_limit_limit": 60,
    "rate_limit_window_seconds": 60,
    "rate_limit_pacing_strategy": "fixed",
    "rate_limit_pacing_start_threshold": 0, # Start pacing immediately
})
async def sla_critical_task(): ...
```

### 3. Queue Management

These settings control behavior when tasks arrive faster than the rate limit allows.

*   **`rate_limit_max_queue_size`** (int | None): The maximum number of tasks permitted to wait for execution. If `None`, the queue is unbounded.
    *   *Default:* `None`
*   **`rate_limit_reject_when_full`** (bool): If `True`, tasks arriving when the queue is full will be rejected immediately with a `RateLimitQueueFullError`.
    *   *Default:* `False`

```python
# Example: Fail-fast configuration.
# If 10 tasks are already waiting, reject subsequent tasks immediately.
@broker.task(labels={
    "rate_limit_enabled": True,
    "rate_limit_limit": 10,
    "rate_limit_max_queue_size": 10,
    "rate_limit_reject_when_full": True,
})
async def realtime_request(): ...
```

### 4. Advanced

*   **`rate_limit_jitter_ms`** (float): Adds up to this many milliseconds of random delay to each paced task. This is useful in distributed environments to prevent workers from executing in perfect lock-step.
    *   *Default:* `0`

```python
# Add up to 50ms of random jitter to task start times.
@broker.task(labels={"rate_limit_enabled": True, "rate_limit_jitter_ms": 50})
async def distributed_task(): ...
```

---

## Deep Dive: Pacing Strategies

The `rate_limit_pacing_strategy` is a key configuration option that fundamentally alters how the limiter schedules tasks after the initial burst (`pacing_start_threshold`) is consumed.

| Strategy  | Primary Goal              | How it Works                                                                       | Best For                                                                                                  |
| :-------- | :------------------------ | :--------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `adaptive`| **Load Smoothing**        | Dynamically calculates delay based on *remaining time* and *remaining slots* in the window. | Protecting downstream services from sudden traffic spikes. Use when avoiding overload is the top priority. |
| `fixed`   | **Throughput Maximization** | Calculates a single, fixed interval (`window / limit`) and schedules tasks at this pace. | Meeting SLAs or processing a predictable volume of tasks. Use when guaranteeing capacity is the top priority. |

### `adaptive` Pacing (Default)

-   **Goal:** To prevent a sudden spike of tasks from overwhelming a system.
-   **Mechanism:** It calculates the spacing between tasks dynamically based on the **time remaining** and **slots remaining** in the current window.
-   **Behavior:** This strategy schedules tasks relative to when the *previous* task was scheduled. A large burst of tasks arriving early can schedule tasks for the entire window's capacity, potentially resulting in **fewer tasks being executed than the `limit`** if the window ends before all scheduled tasks can run.

[rate_limiter_pacing_50_adaptive.mp4](https://github.com/user-attachments/assets/ce4f4a51-3e2e-40a3-8262-c88880fbd433)

### `fixed` Pacing

-   **Goal:** To ensure a predictable number of tasks can be processed in a given window.
-   **Mechanism:** It calculates a single, fixed pace based on the **total window time** and the **total limit** (e.g., `60 seconds / 100 tasks = 0.6s per task`). It then schedules each task at its ideal, pre-calculated time slot.
-   **Behavior:** This strategy guarantees that all `limit` slots are available and spaced evenly throughout the window. If a task arrives *before* its ideal time slot, it will be delayed until that time, ensuring a perfectly even execution rate.

[ rate_limiter_pacing_50_fixed.mp4](https://github.com/user-attachments/assets/12bb3937-1f40-4d50-bd5d-83c7b80c455c) 


### Use Case: No-Burst Pacing (`pacing_start_threshold=0`)

Setting `pacing_start_threshold` to `0` removes the initial burst allowance. This forces the limiter to apply its pacing strategy to every task from the beginning of the window, enabling more granular control over the execution rate.

#### No Burst with `adaptive` Pacing
-   **Behavior:** This configuration enforces load smoothing immediately. The very first task will be subject to a calculated delay to spread execution over the available window. The delay between subsequent tasks remains dynamic.
-   **Use Case:** Ideal for extremely sensitive downstream systems where even a small initial burst of traffic is undesirable. It ensures the smoothest possible ramp-up of tasks.

[rate_limiter_no_burst_adaptive.mp4](https://github.com/user-attachments/assets/293c56ca-8393-425c-b336-794d8f9d479b)

#### No Burst with `fixed` Pacing
-   **Behavior:** This configuration establishes a strict, metronomic execution rate. Every task is delayed to fit into its pre-calculated, evenly spaced time slot (e.g., one task is executed precisely every `0.6` seconds for a `100/60s` limit).
-   **Use Case:** Essential for meeting strict Service Level Agreements (SLAs) that require a constant, predictable processing rate. This guarantees that the system never exceeds the target throughput at any point within the window.

[rate_limiter_no_burst_fixed.mp4](https://github.com/user-attachments/assets/cc546806-91d6-469c-b37a-c9a55c11de88)

## Full Example

This example demonstrates how to configure different tasks with different rate-limiting strategies.

```python
import asyncio
from taskiq import InMemoryBroker
from rate_limit_middleware import RateLimitMiddleware

# Use the default 'adaptive' strategy globally for safety
broker = InMemoryBroker().with_middlewares(RateLimitMiddleware())

# This task targets a sensitive, external API. We want to smooth out calls.
# It will inherit the global 'adaptive' strategy.
@broker.task(
    labels={
        "rate_limit_enabled": True,
        "rate_limit_limit": 50,
        "rate_limit_window_seconds": 60,
        "rate_limit_pacing_start_threshold": 10,
    }
)
async def call_sensitive_api(payload: dict):
    print("Calling sensitive API with adaptive pacing...")
    await asyncio.sleep(0.1)


# This task is for high-volume internal data processing. We want to maximize throughput.
# We override the global default and set the strategy to 'fixed'.
@broker.task(
    labels={
        "rate_limit_enabled": True,
        "rate_limit_limit": 200,
        "rate_limit_window_seconds": 60,
        "rate_limit_pacing_start_threshold": 50,
        "rate_limit_pacing_strategy": "fixed",
    }
)
async def process_internal_data(item_id: int):
    print(f"Processing item {item_id} at max throughput.")
    await asyncio.sleep(0.1)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
