# Copier Template Variables

## Context

All template variables are declared in `copier.yml`. When a user runs
`copier copy`, Copier prompts for each variable and makes them available
as Jinja2 variables in all `.jinja` files under `copier-python-template/`.
Never hardcode values that should vary per project — use variables.

## Invariants

1. All user-facing variables must be declared in `copier.yml` with `type` and `help`. Provide `default` where appropriate.
2. `package_name_py` must always be a valid Python identifier — the default transformation (`lower|replace('-','_')|replace(' ','_')`) handles this automatically.
3. `doc_dir` controls the documentation directory name — the `{{doc_dir}}` folder in the template becomes its value (default: `docs/`).
4. `docker` is the only boolean variable; use `{% if docker %}...{% endif %}` for optional sections.
5. When adding a new variable, update this file's Variable Reference table and all `.jinja` files that use it.

## Variable Reference

| Variable | Type | Default | Used In |
|---|---|---|---|
| `project_name` | str | *(required)* | `settings.py.jinja` (ClearmlSettings.project), `README.md.jinja` |
| `package_name_py` | str | slugified `project_name` | `pyproject.toml.jinja`, `tests/__init__.py.jinja`, `tests/test_zz_*.py.jinja`, `{{ _copier_conf.answers_file }}.jinja` |
| `project_description` | str | *(required)* | `pyproject.toml.jinja`, `README.md.jinja` |
| `python_version` | str (choice) | *(required)* | `pyproject.toml.jinja` (`[tool.poetry.dependencies]`) |
| `author_name` | str | `Name.Surname` | `pyproject.toml.jinja`, `AUTHORS.tmpl` |
| `author_email` | str | `{author_name}@mail.com` | `pyproject.toml.jinja` |
| `repository_provider` | str | *(required)* | `pyproject.toml.jinja`, `{{doc_dir}}/DEVELOPER_GUIDE.md.jinja` |
| `project_namespace` | str | *(required)* | `pyproject.toml.jinja`, `{{doc_dir}}/DEVELOPER_GUIDE.md.jinja` |
| `copyright_date` | str | current year | used in license/copyright blocks |
| `doc_dir` | str | `docs/` | controls `{{doc_dir}}/` directory name; `pyproject.toml.jinja` |
| `docker` | bool | `false` | conditional sections in `pyproject.toml.jinja` |

## How It Works

Copier reads `copier.yml`, prompts the user for each variable without a default,
then renders all `.jinja` files substituting `{{ variable_name }}` with the
provided value. The rendered files are written to the destination directory
without the `.jinja` suffix.

## Agent Checklist

Before adding a new template variable:
- [ ] Is it declared in `copier.yml` with `type`, `help`, and `default` (where applicable)?
- [ ] Is it added to the Variable Reference table in this file?
- [ ] Are all `.jinja` files that use it listed in the table?
- [ ] If boolean, is `{% if variable %}` used for gating optional content?
