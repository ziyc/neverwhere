ALL_ENVS = {}


def add_env(env_id, entrypoint, kwargs):
    if env_id in ALL_ENVS:
        raise RuntimeError(f"environment {env_id} is already registered")
    ALL_ENVS[env_id] = {"entry_point": entrypoint, "kwargs": kwargs}


def list_envs(pattern=None):
    import neverwhere.tasks.scenes  # noqa: F401  registers every scene on first use

    ids = sorted(ALL_ENVS)
    return [i for i in ids if pattern in i] if pattern else ids


def make(env_id, **kwargs):
    import neverwhere.tasks.scenes  # noqa: F401

    env_spec = ALL_ENVS.get(env_id)
    if env_spec is None:
        raise ModuleNotFoundError(f"environment {env_id} is not registered; see neverwhere.list_envs()")
    return env_spec["entry_point"](**{**env_spec["kwargs"], **kwargs})
