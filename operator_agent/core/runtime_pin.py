"""Deploy-time runtime pin: the one place that says which torch the far side gets.

Why a separate module instead of a literal in either caller: this string has two
readers in two layers. `decide.reference_contract.PINNED_RUNTIME` is the *identity*
the reference gate compares against; `execute.remote_executor._setup_uv_env` is the
*install* step. Layers go `app → generate → decide → execute → core`, so `execute`
cannot import `decide` — the single source lives here, and both read downward.

What went wrong without it (2026-11-08, H20): the deploy template ran
``uv add numpy torch triton`` with no version constraint, so a work directory
created earlier at torch 2.14.0 was silently upgraded to 2.14.1 on the next deploy.
The identity gate then failed closed (``native_libraries, torch_config_sha256,
torch_git, torch_version``) and every trusted-reference task — attention and
layernorm v2 — stopped before L2. Pinning the *requirement* and re-reading the
installed version is the whole fix; the gate stays as strict as it was.
"""

#: Attested build, byte-for-byte the value `PINNED_RUNTIME` publishes. Includes the
#: local version tag because that is what `torch.__version__` reports.
TORCH_VERSION = "2.14.0+cu130"


def torch_requirement() -> str:
    """The requirement string for ``uv add``: the *public* version, no local tag.

    PyPI/mirror wheels are named ``torch-2.14.0-cp310-...whl`` and carry the CUDA tag
    inside the build, so the requirement is ``torch==2.14.0`` while the installed
    ``torch.__version__`` is ``2.14.0+cu130``. Two different strings, one build:
    resolving to ``2.14.0`` is *not* evidence of identity — the caller must read the
    version back after installing (see ``remote_executor._setup_uv_env``).
    """
    public = TORCH_VERSION.split("+", 1)[0].strip()
    if not public:
        raise ValueError("runtime pin is empty: refusing to deploy an unpinned torch")
    return f"torch=={public}"


def matches(observed: str) -> bool:
    """Whether an observed ``torch.__version__`` is the attested build."""
    return isinstance(observed, str) and observed.strip() == TORCH_VERSION