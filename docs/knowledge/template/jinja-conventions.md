# Jinja Template Conventions

## Context

Copier uses Jinja2 to render template files. Only files with the `.jinja` suffix
are rendered — plain files are copied verbatim. The template root is
`copier-python-template/` (set via `_subdirectory` in `copier.yml`), not the
repo root. This means all paths in `copier.yml` config keys are relative to
`copier-python-template/`.

## Invariants

1. Only `.jinja` files are rendered. To template a file, rename it to `<name>.jinja`. The `.jinja` suffix is stripped in the output (e.g., `settings.py.jinja` → `settings.py`).
2. Template root is `copier-python-template/` — the repo root is never the template source.
3. Files listed in `_skip_if_exists` in `copier.yml` are **never overwritten** on `copier update`. Use this for files users are expected to customize (`pyproject.toml`, `__init__.py`, `README.md`).
4. Standard Jinja2 delimiters apply: `{{ variable }}` for substitution, `{% if/for/block %}` for logic, `{# comment #}` for comments.
5. Directory names can also be templated: `{{doc_dir}}/` becomes the value of the `doc_dir` variable (e.g., `docs/`). Use this pattern for optional or configurable directories.
6. `_tasks` in `copier.yml` run shell commands after rendering. Keep tasks idempotent (guarded by `git log &>/dev/null ||`).

## How to Add a New Templated File

1. Create the file in `copier-python-template/` at the desired output path.
2. Add `.jinja` suffix: `src/myconfig.py` → `src/myconfig.py.jinja`.
3. Use `{{ variable_name }}` for substitutions from variables declared in `copier.yml`.
4. Use `{% if condition %}...{% endif %}` for sections gated on boolean variables.
5. If users should customize this file after generation (not have it overwritten on `copier update`), add its path to `_skip_if_exists` in `copier.yml`.

## How to Add a New Plain (Non-Templated) File

1. Create the file in `copier-python-template/` without `.jinja` suffix.
2. Do not add it to any special config — Copier copies it verbatim.

## Agent Checklist

Before adding or modifying a template file:
- [ ] Does the file have `.jinja` suffix if it needs variable substitution?
- [ ] Are only variables declared in `copier.yml` used inside `{{ }}`?
- [ ] If user-customizable, is its path in `_skip_if_exists` in `copier.yml`?
- [ ] If adding a variable-named directory, does the folder use `{{variable}}` syntax?
- [ ] If adding a `_tasks` entry, is it guarded against re-running on existing repos?
