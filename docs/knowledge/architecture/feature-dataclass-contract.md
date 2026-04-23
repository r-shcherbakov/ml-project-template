# Feature Dataclass Contract

## Context

Features are defined as frozen dataclasses in `src/common/features.py`.
`MANDATORY_FEATURES` is the single source of truth — any change here
automatically propagates to clip configs, fillna configs, and dtype configs
in `src/common/config.py`. This eliminates manual synchronization across files.

## Invariants

1. `Feature` is `frozen=True` — never mutate instances. Never use `dataclasses.replace()` to work around immutability.
2. `MANDATORY_FEATURES` in `features.py` is the only place to register features that need clip/fillna/dtype enforcement. Do not add them anywhere else.
3. Required fields for any `Feature` used in `MANDATORY_FEATURES`: `name` (str) and `dtype` (str). All other fields (`lower`, `upper`, `fillna_value`, `fillna_method`, `fillna_limit`) are optional.
4. `IGNORED_FEATURES` lists features present in the data but excluded from processing (e.g., `GROUP_ID`, grouping columns, ID columns). Add such columns here, not to `MANDATORY_FEATURES`.
5. Never edit `src/common/config.py` directly — `FEATYPE_TYPES`, `CLIP_CONFIG`, and `FILLNA_CONFIG` are derived from `MANDATORY_FEATURES` at module load time.

## How It Works

```python
# features.py — single source of truth
MANDATORY_FEATURES = [FEATURE_1, FEATURE_2, FEATURE_3]

# config.py — auto-derived at import time, never edit manually
FEATYPE_TYPES = {f.name: f.dtype for f in MANDATORY_FEATURES}

CLIP_CONFIG = {
    f.name: {"lower": f.lower, "upper": f.upper}
    for f in MANDATORY_FEATURES
}

FILLNA_CONFIG = {
    f.name: {"value": f.fillna_value, "method": f.fillna_method, "limit": f.fillna_limit}
    for f in MANDATORY_FEATURES
}
```

To add a new feature that needs processing:
```python
# 1. Define in features.py
MY_FEATURE = Feature(
    name="my_feature",
    dtype="float32",
    lower=0.0,
    upper=100.0,
    fillna_value=0.0,
)

# 2. Add to MANDATORY_FEATURES (config.py updates automatically)
MANDATORY_FEATURES = [FEATURE_1, FEATURE_2, FEATURE_3, MY_FEATURE]
```

## Agent Checklist

Before adding or modifying a feature:
- [ ] Is the `Feature` instance defined in `src/common/features.py`?
- [ ] Is it added to `MANDATORY_FEATURES` (if it needs clip/fillna/dtype enforcement)?
- [ ] Does it have both `name` and `dtype` set?
- [ ] Is `src/common/config.py` left untouched (configs regenerate automatically)?
- [ ] If it's a grouping/ID column, is it in `IGNORED_FEATURES` instead?
