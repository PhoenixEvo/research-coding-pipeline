# Research Experiment Pipeline

A reusable LangGraph workflow for iterative AI/ML experimentation. Swap a YAML
file to move between projects such as 3D Gaussian Splatting compression, speech
emotion recognition, and NLP hyperparameter studies.

The graph follows this loop:

```text
generate configs -> run experiments -> analyze logs -> optimize
       ^                                             |
       |---------------- continue -------------------|
                                                     |
                                      stop -> final report
```

Each node persists `state.json`. Individual experiment failures become result
records, so a failed remote run does not terminate the whole pipeline.

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. Telegram is optional. Kaggle can use
`KAGGLE_USERNAME` and `KAGGLE_KEY`, or the standard
`~/.kaggle/kaggle.json` file.

Run an example:

```powershell
uv run research-pipeline configs/3dgs_compression.yaml
```

Resume from the node after the last successful snapshot:

```powershell
uv run research-pipeline configs/3dgs_compression.yaml `
  --resume outputs/3dgs_compression/state.json
```

Run local tests without API keys:

```powershell
uv run pytest
```

## Configuration

The main reusable sections are:

- `metrics`: names and `maximize`/`minimize` directions.
- `baselines`: comparison methods and their metric values.
- `search_space`: lists or numeric ranges for training parameters.
- `experiments_per_iteration`, `max_experiments`, `max_iterations`, and
  `time_budget_hours`: stopping budgets.
- `improvement_threshold`: minimum primary-metric gain. The graph stops after
  two consecutive iterations below this threshold.
- `platform`: initial executor, `kaggle` or `modal`.
- `llm`: Gemini models, fallback model, temperature, and whether calls are enabled.

Set `llm.enabled: false` for deterministic local candidate generation and
statistical analysis. This mode is useful for CI and executor development.

All LLM calls use Gemini through `langchain-google-genai`:

```yaml
llm:
  provider: google
  fast_model: gemini-3.2-flash
  analysis_model: gemini-3.5-flash
  fallback_model: gemini-3.5-flash
  temperature: 0.2
```

Candidate generation and LLM-assisted error diagnosis use `fast_model`.
`log_analyst` and `report_generator` use `analysis_model`. If the fast model
returns a model-not-found response, the call retries once with `fallback_model`.
The fallback exists because Gemini endpoint availability can vary by account
and region.

## Remote executor contracts

### Kaggle

Set `kaggle.kernel_source_dir` to a directory containing the configured
`code_file`, normally `kernel.py`. Before submission, the executor copies that
directory and adds `experiment_config.yaml`. The training script should read
that file and write one of:

- the configured `kaggle.metrics_file`;
- `metrics.json` or `results.json`;
- `metrics.csv` or `results.csv`; or
- named `metric=value` entries in a downloaded text/log file.

The executor creates `kernel-metadata.json`, pushes the kernel, polls it, and
downloads outputs. Kaggle quota errors automatically fall back to Modal when
Modal is configured. OOM failures retry once with half the batch size.

### Modal

Configure an existing deployed function:

```yaml
modal:
  app_name: research-experiments
  function_name: run_experiment
  timeout_seconds: 28800
```

The function receives `(exp_config, project_config)` and must return a
dictionary containing at least:

```python
{
    "metrics": {"accuracy": 0.84},
    "status": "completed",
}
```

The pipeline fills missing ID, platform, duration, config, and error fields.

## Outputs

Every project writes to its configured `output_dir`:

- `state.json`: atomic crash-recovery snapshot;
- `results.json`: normalized experiment records;
- `final_report.md`: executive summary, best config, LaTeX table, baseline
  comparison, ablation ideas, methodology findings, and limitations;
- metric tradeoff, sensitivity, baseline, and optional training-curve charts.

Plotly attempts PNG export and falls back to HTML if a local image export
backend is unavailable.

## Project layout

```text
pipeline.py              LangGraph assembly, CLI, persistence, resume
state.py                 Typed state and normalized result schemas
nodes/                   generation, execution, analysis, stopping, reporting
executors/               Kaggle and Modal integrations
utils/                   LLMs, retries, metrics, notifications, charts
configs/                 reusable project YAML files
tests/                   mocked end-to-end and recovery tests
```
