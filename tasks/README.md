# tasks/

Local, project-only Isaac Lab gym task definitions — one subfolder per task.
Not part of the `isaaclab_tasks` package; these live here because R1 is our
own robot, not something upstream Isaac Lab ships a task for.

Currently: `r1_flat/` (R1 flat-ground velocity-tracking task, Week02).

## Why a separate package instead of patching Isaac Lab

Isaac Lab's own scripts (`train.py`, `random_agent.py`, etc.) only
`import isaaclab_tasks` to trigger built-in task registration — they have no
way to know about a task package living outside the IsaacLab install. Rather
than edit the shared IsaacLab checkout (not ours to modify, not tracked in
this repo), any script that needs our tasks registered does it itself:

```python
sys.path.insert(0, str(_PROJECT_ROOT))
import tasks.r1_flat  # noqa: F401  -- runs the gym.register() calls
```

See `scripts/random_agent_r1.py` for a working example, and its docstring for
why we couldn't just reuse Isaac Lab's own `random_agent.py`.
