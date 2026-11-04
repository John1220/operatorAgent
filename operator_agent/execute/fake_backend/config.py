"""Closed, serializable test scenarios; no device or provider auto-detection."""
from dataclasses import asdict, dataclass


SCENARIOS = ("success", "compile-repair", "correctness-repair", "no-improvement",
             "timeout", "transport-unknown", "hardware-missing")
PROFILES = ("small", "large", "unknown")


@dataclass(frozen=True)
class FakeConfig:
    scenario: str = "success"
    profile: str = "small"
    llm: str = "fake"

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError("unknown fake scenario: " + str(self.scenario))
        if self.profile not in PROFILES:
            raise ValueError("unknown fake profile: " + str(self.profile))
        if self.llm not in ("fake", "live"):
            raise ValueError("fake llm must be fake or live")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value=None):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid fake configuration")
        return cls(**value)


def execution_context(config):
    """Attach to launch, status and artifacts, including interrupted runs."""
    config = FakeConfig.from_dict(config) if not isinstance(config, FakeConfig) else config
    return dict(mode="fake", evidence_kind="test_only", test_only=True,
                gpu_acceptance=False, performance_evidence=False,
                model_mode=config.llm, fake=config.as_dict())


def tool_context(backend, *, model_mode='not_used'):
    """Foreground tools have no scenario options; only repair uses a model."""
    if backend != 'fake':
        return {}
    context = execution_context(FakeConfig())
    context.pop('fake')
    context['model_mode'] = model_mode
    return dict(execution_context=context)
