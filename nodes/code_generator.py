"""Generate candidate experiment configurations."""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from typing import Any

from nodes.common import best_result, update_state
from state import ResearchState
from utils.llm import invoke_json

LOGGER = logging.getLogger(__name__)


def _space_values(spec: Any) -> list[Any]:
    """Expand one YAML search-space declaration into candidate values."""

    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        return [spec]
    if "values" in spec:
        return list(spec["values"])
    low = spec.get("min")
    high = spec.get("max")
    if low is None or high is None:
        return [spec.get("default")]
    count = max(2, int(spec.get("samples", 5)))
    if spec.get("type") == "int":
        if count == 1:
            return [int(low)]
        values = {
            round(float(low) + index * (float(high) - float(low)) / (count - 1))
            for index in range(count)
        }
        return sorted(values)
    if spec.get("scale") == "log":
        import math

        low_log, high_log = math.log10(float(low)), math.log10(float(high))
        return [
            10 ** (low_log + index * (high_log - low_log) / (count - 1))
            for index in range(count)
        ]
    return [
        float(low) + index * (float(high) - float(low)) / (count - 1)
        for index in range(count)
    ]


def _deterministic_candidates(state: ResearchState, count: int) -> list[dict[str, Any]]:
    """Generate diverse reproducible candidates without an LLM dependency."""

    project_config = state["project_config"]
    space = project_config.get("search_space", {})
    names = list(space)
    value_sets = [_space_values(space[name]) for name in names]
    products = [dict(zip(names, values)) for values in itertools.product(*value_sets)]
    rng = random.Random(int(project_config.get("random_seed", 42)) + state["current_iteration"])
    rng.shuffle(products)

    previous = {
        json.dumps(
            {key: value for key, value in result.get("config", {}).items() if not key.startswith("_")},
            sort_keys=True,
            default=str,
        )
        for result in state["experiment_results"]
    }
    selected: list[dict[str, Any]] = []
    best = best_result(state)
    if best and state["current_iteration"] > 0:
        base = {
            key: value
            for key, value in best["config"].items()
            if key in space
        }
        for index, name in enumerate(names):
            for value in value_sets[index]:
                candidate = dict(base)
                candidate[name] = value
                signature = json.dumps(candidate, sort_keys=True, default=str)
                if signature not in previous:
                    selected.append(candidate)
                    previous.add(signature)
                if len(selected) >= max(1, count // 2):
                    break
            if len(selected) >= max(1, count // 2):
                break

    for candidate in products:
        signature = json.dumps(candidate, sort_keys=True, default=str)
        if signature not in previous:
            selected.append(candidate)
            previous.add(signature)
        if len(selected) >= count:
            break
    return selected[:count]


def _llm_candidates(state: ResearchState, count: int) -> list[dict[str, Any]]:
    """Ask the configured model for candidates constrained to the search space."""

    config = state["project_config"]
    prompt = (
        f"Generate {count} experiment configurations for iteration "
        f"{state['current_iteration'] + 1}. Stay within the YAML search space. "
        "Use previous results to balance exploitation with exploration. "
        "Return an object with an `experiments` array containing only hyperparameters.\n\n"
        f"Search space:\n{json.dumps(config.get('search_space', {}), indent=2)}\n\n"
        f"Previous results:\n{json.dumps(state['experiment_results'][-20:], indent=2)}\n\n"
        f"Improvement suggestions:\n{json.dumps(state['improvement_suggestions'], indent=2)}"
    )
    payload = invoke_json(prompt, config.get("llm", {}), tier="fast")
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("LLM response `experiments` must be a list")
    allowed = set(config.get("search_space", {}))
    return [
        {key: value for key, value in item.items() if key in allowed}
        for item in experiments
        if isinstance(item, dict)
    ][:count]


def code_generator(state: ResearchState) -> ResearchState:
    """Generate and identify the next batch of experiment configs."""

    started = time.monotonic()
    config = state["project_config"]
    iteration = state["current_iteration"] + 1
    count = int(config.get("experiments_per_iteration", 3))
    remaining = max(
        0,
        int(config.get("max_experiments", count)) - len(state["experiment_results"]),
    )
    count = min(count, remaining)
    LOGGER.info(
        "code_generator start: iteration=%s requested=%s previous_results=%s",
        iteration,
        count,
        len(state["experiment_results"]),
    )

    candidates: list[dict[str, Any]] = []
    llm_config = config.get("llm", {})
    if count and llm_config.get("enabled", True):
        try:
            LOGGER.info("code_generator invoking LLM")
            candidates = _llm_candidates(state, count)
        except Exception:
            LOGGER.exception("LLM candidate generation failed; using deterministic search")
    if len(candidates) < count:
        fallback = _deterministic_candidates(state, count * 2)
        signatures = {json.dumps(item, sort_keys=True, default=str) for item in candidates}
        for candidate in fallback:
            signature = json.dumps(candidate, sort_keys=True, default=str)
            if signature not in signatures:
                candidates.append(candidate)
                signatures.add(signature)
            if len(candidates) >= count:
                break

    platform = str(config.get("platform", "kaggle"))
    identified = [
        {
            **candidate,
            "id": f"iter-{iteration:02d}-exp-{index:03d}",
            "_iteration": iteration,
            "_platform": platform,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    LOGGER.info(
        "code_generator complete: generated=%s duration=%.3fs",
        len(identified),
        time.monotonic() - started,
    )
    return update_state(
        state,
        current_iteration=iteration,
        experiment_configs=identified,
    )
