# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "taskiq_rate_limiter",
# ]
# ///

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from taskiq_rate_limiter.rate_limit_middleware import *

# OPTIMIZATION PARAMETERS
FRAME_SKIP_RATE = 1
DPI_RENDER = 150
DATA_CAPTURE_INTERVAL = 0.1


# --- Data Structures and Helpers ---
@dataclass
class SimulationState:
    time: float
    window_start: Optional[float]
    window_end: Optional[float]
    granted_in_window: int
    next_allowed_time: Optional[float]
    pacing_threshold: int
    pending_count: int
    waiting_count: int
    running_count: int
    done_count: int


class VisualizableTaskStartRateLimiter(TaskStartRateLimiter):
    def get_internal_state(self) -> Dict[str, Any]:
        return {
            "window_start": self._window_start, "window_end": self._window_end,
            "granted_in_window": self._granted_in_window, "next_allowed_time": self._next_allowed_time,
            "pacing_threshold": self._pacing_threshold,
        }


async def run_simulation(
    config: RateLimitConfig,
    num_tasks: int,
    arrival_window_seconds: float,
    task_duration_seconds: float,
) -> List[SimulationState]:
    """Runs the rate limiter simulation and returns the history of states."""
    print("Starting simulation to gather data...")
    # --- State variables are now local to the simulation run ---
    simulation_history: List[SimulationState] = []
    tasks_status: List[str] = ["PENDING"] * num_tasks
    simulation_start_time: float = 0.0

    limiter = VisualizableTaskStartRateLimiter(config=config)

    # --- Helper functions are nested to capture local state ---
    async def task_worker(task_id: int, limiter_instance: VisualizableTaskStartRateLimiter):
        """A single task that gets throttled by the rate limiter."""
        nonlocal tasks_status  # Explicitly state we are modifying the outer scope variable
        tasks_status[task_id] = "WAITING"
        await limiter_instance.throttle(f"task-{task_id}")
        tasks_status[task_id] = "RUNNING"
        await asyncio.sleep(task_duration_seconds)
        tasks_status[task_id] = "DONE"

    async def data_recorder(limiter_instance: VisualizableTaskStartRateLimiter, all_tasks: List[asyncio.Task]):
        """Periodically records the state of the simulation."""
        nonlocal simulation_history, simulation_start_time
        while not all(t.done() for t in all_tasks):
            now = time.monotonic() - simulation_start_time
            limiter_state = limiter_instance.get_internal_state()
            window_start = (limiter_state["window_start"] - simulation_start_time) if limiter_state[
                "window_start"
            ] else None
            window_end = (limiter_state["window_end"] - simulation_start_time) if limiter_state["window_end"] else None
            next_allowed = (limiter_state["next_allowed_time"] - simulation_start_time) if limiter_state[
                "next_allowed_time"
            ] else None
            state = SimulationState(
                time=now, window_start=window_start, window_end=window_end,
                granted_in_window=limiter_state["granted_in_window"], next_allowed_time=next_allowed,
                pacing_threshold=limiter_state["pacing_threshold"], pending_count=tasks_status.count("PENDING"),
                waiting_count=tasks_status.count("WAITING"), running_count=tasks_status.count("RUNNING"),
                done_count=tasks_status.count("DONE"),
            )
            simulation_history.append(state)
            await asyncio.sleep(DATA_CAPTURE_INTERVAL)

    # --- Simulation Execution ---
    simulation_start_time = time.monotonic()

    task_coroutines = []
    for i in range(num_tasks):
        delay = random.uniform(0, arrival_window_seconds)

        async def delayed_start(task_id=i, start_delay=delay):
            await asyncio.sleep(start_delay)
            await task_worker(task_id, limiter)

        task_coroutines.append(asyncio.create_task(delayed_start()))

    recorder_task = asyncio.create_task(data_recorder(limiter, task_coroutines))
    await asyncio.gather(*task_coroutines)
    await asyncio.sleep(0.1)
    recorder_task.cancel()
    print(f"Simulation finished. Collected {len(simulation_history)} data points.")

    return simulation_history


def create_animation(
    simulation_history: List[SimulationState],
    title: str,
    file_location: str,
    num_tasks: int,
    config: RateLimitConfig,
):
    """Creates and saves the animation from the simulation history."""
    if not simulation_history:
        print("No simulation data to animate.")
        return

    history_to_render = simulation_history[::FRAME_SKIP_RATE]
    if not history_to_render:
        print("Error: Rendering history is empty after skipping frames.")
        return

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    times = [s.time for s in history_to_render]
    next_allowed_times = [s.next_allowed_time for s in history_to_render]
    total_duration = times[-1]

    valid_next_allowed = [t for t in next_allowed_times if t is not None and not np.isnan(t)]
    max_y_limit = max(valid_next_allowed) * 1.1 if valid_next_allowed else total_duration
    max_y_limit = max(max_y_limit, 1.0)

    # --- Plot 3: Pacing Timeline Setup ---
    ax3.set_title(title)
    ax3.set_xlabel("Simulation Time (s)")
    ax3.set_ylabel("Scheduled Start Time (s)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.plot([0, total_duration], [0, total_duration], "k--", alpha=0.7, label="Immediate Start (No Delay)")

    plot_next_allowed = [t if t is not None else np.nan for t in next_allowed_times]
    pacing_line, = ax3.plot([], [], "b-", lw=2.5, label="Next Allowed Start Time")
    now_vline = ax3.axvline(0, color="r", linestyle="-", lw=2, alpha=0.8, label="Current Time")

    ax3.axvspan(0, 0, color="gray", alpha=0.2, label=f"Current {config.window_seconds}s Window")
    window_span_container = []

    ax3.set_xlim(0, total_duration)
    ax3.set_ylim(0, max_y_limit)
    ax3.legend(loc="upper left")

    def update(frame: int):
        state = history_to_render[frame]
        now = state.time
        print(f"Rendering frame {frame+1}/{len(history_to_render)}...", end="\r")

        # --- Plot 1: Task Status ---
        ax1.clear()
        ax1.set_title(f"Task Status @ {now:.2f}s")
        statuses = ["Done", "Running", "Waiting", "Pending"]
        counts = [state.done_count, state.running_count, state.waiting_count, state.pending_count]
        colors = ["#4CAF50", "#2196F3", "#FFC107", "#9E9E9E"]
        bars = ax1.barh(statuses, counts, color=colors)
        ax1.set_xlim(0, num_tasks * 1.1)
        ax1.bar_label(bars, padding=3)
        ax1.set_xlabel("Number of Tasks")

        # --- Plot 2: Window Rate Limit ---
        ax2.clear()
        ax2.set_title("Current Window Status")
        ax2.set_xlim(0, config.limit)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("Granted Permits / Limit")
        ax2.barh([0.5], [state.granted_in_window], height=0.2, color="#2196F3", edgecolor="black")
        ax2.text(config.limit / 2, 0.75, f"{state.granted_in_window} / {config.limit}", ha="center", fontsize=12, weight="bold")
        pacing_thresh = state.pacing_threshold
        ax2.axvline(pacing_thresh, color="orange", linestyle="--", lw=2, label=f"Pacing Threshold ({pacing_thresh})")
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))

        # --- Plot 3: Pacing Timeline Update ---
        now_vline.set_xdata([now, now])
        pacing_line.set_data(times[:frame + 1], plot_next_allowed[:frame + 1])
        if window_span_container:
            window_span_container.pop().remove()
        if state.window_start is not None and state.window_end is not None:
            span = ax3.axvspan(state.window_start, state.window_end, color="gray", alpha=0.2)
            window_span_container.append(span)

    num_frames_to_render = len(history_to_render)
    print("\n--- Animation Optimization Summary ---")
    print(f"Total data points collected: {len(simulation_history)}")
    print(f"Frames to render (after skip rate {FRAME_SKIP_RATE}): {num_frames_to_render}")
    print(f"Target FPS: 20, Target DPI: {DPI_RENDER}")
    print("--------------------------------------")
    print("Creating and saving animation...")

    ani = FuncAnimation(fig, update, frames=num_frames_to_render, blit=False, interval=50)
    ani.save(file_location, writer="ffmpeg", fps=20, dpi=DPI_RENDER)
    print(f"\nAnimation saved successfully as {file_location}")


def main():
    """Defines parameters and runs the simulation and animation for multiple configurations."""
    num_tasks = 200
    media_dir = "../docs/media/"
    # A list of configurations for the different videos
    configurations = [
        {
            "simulation_config": {"limit": 100, "window_seconds": 60, "pacing_start_threshold": 0, "pacing_strategy": "fixed"},
            "file_location": f"{media_dir}rate_limiter_no_burst_fixed.mp4",
        },
        {
            "simulation_config": {"limit": 100, "window_seconds": 60, "pacing_start_threshold": 0, "pacing_strategy": "adaptive"},
            "file_location": f"{media_dir}rate_limiter_no_burst_adaptive.mp4",
        },
        {
            "simulation_config": {"limit": 100, "window_seconds": 60, "pacing_start_threshold": 50, "pacing_strategy": "fixed"},
            "file_location": f"{media_dir}rate_limiter_pacing_50_fixed.mp4",
        },
        {
            "simulation_config": {"limit": 100, "window_seconds": 60, "pacing_start_threshold": 50, "pacing_strategy": "adaptive"},
            "file_location": f"{media_dir}rate_limiter_pacing_50_adaptive.mp4",
        },
    ]

    for config_details in configurations:
        simulation_config = config_details["simulation_config"]
        file_location = config_details["file_location"]

        # 1. Run the simulation
        history = asyncio.run(
            run_simulation(
                config=RateLimitConfig(**simulation_config),
                num_tasks=num_tasks,
                arrival_window_seconds=90,
                task_duration_seconds=1,
            ),
        )

        # 2. Create and save the animation if data was generated
        if history:
            create_animation(
                simulation_history=history,
                title=f"Pacing Mechanism Timeline {simulation_config}",
                file_location=file_location,
                num_tasks=num_tasks,
                config=RateLimitConfig(**simulation_config),
            )



if __name__ == "__main__":
    main()
