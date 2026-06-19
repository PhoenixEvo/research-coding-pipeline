# Codex/Antigravity Prompt Bridge

This bridge reads the most recently updated project directory under `outputs/`
and converts its experiment configs, errors, optimizer suggestions, and analysis
into prompts that are ready to paste into Codex or Antigravity. It does not call
an LLM or modify source code.

## Commands

Apply the highest-primary-metric configuration:

```powershell
uv run python bridge/codex_prompt_gen.py apply
```

Example beginning:

```text
# Context: Apply the pipeline's best experiment configuration to the project
Apply the best pipeline configuration below to `configs/3dgs_compression.yaml`.
```

Diagnose the latest experiment error:

```powershell
uv run python bridge/codex_prompt_gen.py fix
```

Example beginning:

```text
# Context: Diagnose and fix the latest failed pipeline experiment
Diagnose and fix the root cause of this failed research experiment.
```

Implement the optimizer's top suggestion:

```powershell
uv run python bridge/codex_prompt_gen.py suggest
```

Example beginning:

```text
# Context: Implement the optimizer's top improvement suggestion
Implement this highest-priority pipeline suggestion:
```

Refactor for the next iteration using the analysis findings:

```powershell
uv run python bridge/codex_prompt_gen.py refactor
```

Example beginning:

```text
# Context: Refactor training code to support the next experiment iteration
Refactor the training implementation to better support the next iteration's focus area.
```

Print all four prompts:

```powershell
uv run python bridge/codex_prompt_gen.py all
```

Each prompt ends with `# ---`. Missing artifacts produce a clear message instead
of a traceback.

## Typical workflow

1. Run the research pipeline.
2. Run the appropriate bridge command.
3. Copy the generated prompt into Codex or Antigravity.
4. Review and apply the agent's changes.
5. Run the pipeline again to measure the next experiment iteration.
