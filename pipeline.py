"""Command-line entry point and LangGraph orchestration."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from executors.kaggle_executor import run_kaggle_experiment
from executors.modal_executor import run_modal_experiment
from nodes.code_generator import code_generator
from nodes.experiment_runner import Executor, experiment_runner
from nodes.log_analyst import log_analyst
from nodes.optimizer import optimizer
from nodes.report_generator import report_generator
from state import ResearchState, initial_state

LOGGER = logging.getLogger(__name__)
Node = Callable[[ResearchState], ResearchState | Awaitable[ResearchState]]


def configure_logging(level: str = "INFO") -> None:
    """Configure timestamped application logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML project configuration."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Project YAML must contain a mapping at its root")
    required = {"name", "metrics", "search_space", "max_experiments"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Project YAML is missing required keys: {', '.join(missing)}")
    output_dir = Path(config.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
        output_dir = (base_dir / output_dir).resolve()
    config["output_dir"] = str(output_dir)
    config["_config_path"] = str(path)
    config.setdefault("_pipeline_started_at", time.time())
    return config


def save_state(state: ResearchState) -> Path:
    """Persist graph state atomically for crash recovery."""

    output_dir = Path(state["project_config"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "state.json"
    temporary = output_dir / "state.json.tmp"
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def load_state(state_path: str | Path) -> ResearchState:
    """Load a previously persisted graph state."""

    with Path(state_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return cast(ResearchState, data)


def _with_persistence(name: str, node: Node) -> Node:
    """Wrap a node so every successful transition is persisted."""

    if inspect.iscoroutinefunction(node):

        async def async_wrapped(state: ResearchState) -> ResearchState:
            updated = await cast(Callable[[ResearchState], Awaitable[ResearchState]], node)(
                state
            )
            started_at = float(
                updated["project_config"].get("_pipeline_started_at", time.time())
            )
            updated["project_config"]["_elapsed_seconds"] = time.time() - started_at
            updated["project_config"]["_last_node"] = name
            path = save_state(updated)
            LOGGER.info("Persisted state after %s to %s", name, path)
            return updated

        return async_wrapped

    def wrapped(state: ResearchState) -> ResearchState:
        updated_or_awaitable = node(state)
        if inspect.isawaitable(updated_or_awaitable):
            raise TypeError(f"Node {name} returned an awaitable from a synchronous wrapper")
        updated = updated_or_awaitable
        started_at = float(
            updated["project_config"].get("_pipeline_started_at", time.time())
        )
        updated["project_config"]["_elapsed_seconds"] = time.time() - started_at
        updated["project_config"]["_last_node"] = name
        path = save_state(updated)
        LOGGER.info("Persisted state after %s to %s", name, path)
        return updated

    return wrapped


def build_graph(
    *,
    kaggle_executor: Executor = run_kaggle_experiment,
    modal_executor: Executor = run_modal_experiment,
) -> Any:
    """Build and compile the reusable research optimization graph."""

    async def runner(state: ResearchState) -> ResearchState:
        return await experiment_runner(
            state,
            kaggle_executor=kaggle_executor,
            modal_executor=modal_executor,
        )

    builder = StateGraph(ResearchState)
    builder.add_node("code_generator", _with_persistence("code_generator", code_generator))
    builder.add_node("experiment_runner", _with_persistence("experiment_runner", runner))
    builder.add_node("log_analyst", _with_persistence("log_analyst", log_analyst))
    builder.add_node("optimizer", _with_persistence("optimizer", optimizer))
    builder.add_node(
        "report_generator", _with_persistence("report_generator", report_generator)
    )
    builder.add_conditional_edges(
        START,
        lambda state: state["project_config"].get("_entry_node", "code_generator"),
        {
            "code_generator": "code_generator",
            "experiment_runner": "experiment_runner",
            "log_analyst": "log_analyst",
            "optimizer": "optimizer",
            "report_generator": "report_generator",
        },
    )
    builder.add_edge("code_generator", "experiment_runner")
    builder.add_edge("experiment_runner", "log_analyst")
    builder.add_edge("log_analyst", "optimizer")
    builder.add_conditional_edges(
        "optimizer",
        lambda state: "continue" if state["should_continue"] else "stop",
        {"continue": "code_generator", "stop": "report_generator"},
    )
    builder.add_edge("report_generator", END)
    return builder.compile()


async def run_pipeline(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    kaggle_executor: Executor = run_kaggle_experiment,
    modal_executor: Executor = run_modal_experiment,
) -> ResearchState:
    """Run a project from YAML or resume a prior state snapshot."""

    if resume_path:
        state = load_state(resume_path)
        last_node = state["project_config"].get("_last_node")
        if last_node == "report_generator":
            return state
        next_nodes = {
            "code_generator": "experiment_runner",
            "experiment_runner": "log_analyst",
            "log_analyst": "optimizer",
            "optimizer": (
                "code_generator"
                if state.get("should_continue", True)
                else "report_generator"
            ),
        }
        state["project_config"]["_entry_node"] = next_nodes.get(
            str(last_node), "code_generator"
        )
        state["project_config"]["_pipeline_started_at"] = time.time() - float(
            state["project_config"].get("_elapsed_seconds", 0.0)
        )
    else:
        state = initial_state(load_project_config(config_path))
    graph = build_graph(
        kaggle_executor=kaggle_executor,
        modal_executor=modal_executor,
    )
    max_iterations = int(state["project_config"].get("max_iterations", 10_000))
    recursion_limit = max(25, max_iterations * 5 + 10)
    result = await graph.ainvoke(
        state,
        config={"recursion_limit": recursion_limit},
    )
    return cast(ResearchState, result)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a project YAML config")
    parser.add_argument("--resume", help="Path to a previously saved state.json")
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI."""

    load_dotenv()
    args = _parse_args()
    configure_logging(args.log_level)
    final_state = asyncio.run(run_pipeline(args.config, resume_path=args.resume))
    report_path = Path(final_state["project_config"]["output_dir"]) / "final_report.md"
    LOGGER.info("Pipeline finished. Final report: %s", report_path)


if __name__ == "__main__":
    main()
