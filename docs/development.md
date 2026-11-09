# Development

ketjet is developed in the open using Scrum with goal-based sprints and a Kanban board.

- **Contributing guide:** setup, checks and workflow are in
  [`CONTRIBUTING.md`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/CONTRIBUTING.md).
- **Roadmap:** the product backlog, sprint plan and quality bar are in
  [`plan.md`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/plan.md).
- **Board:** [GitHub Projects](https://github.com/users/hasanahmed77/projects/2).

## Building these docs

```bash
uv sync --group docs
uv run mkdocs serve         # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict
```
