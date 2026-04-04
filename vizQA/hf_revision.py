"""Resolve which Git revision to use when downloading MiniLM weights from Hugging Face.

The Python package version and the weights repo tag do not have to move in lockstep: a
release may bump only code while the model stays unchanged. This module picks a Hub
revision that is safe to ship with a given *package* version:

1. **Exact match** — If some tag parses to the same version as the package (PEP 440),
   use that tag. Tag names may be ``v0.2.0`` or ``0.2.0``; we compare parsed versions,
   not raw strings, so both normalize consistently after stripping a leading ``v``.

2. **Latest older weights** — If no tag equals the package version, use the tag with
   the **greatest** version that is **strictly less** than the package version. That
   reuses the last published weights for this release line instead of failing CI.

3. **Fallback** — If listing tags fails, no tag parses as a version, or every tag is
   newer than the package version, return ``"main"``. Callers should treat that as
   best-effort; private repos may need a valid ``HF_TOKEN`` for ``list_repo_refs``.

Used by :mod:`vizQA.install` and by release workflows via ``python -m vizQA.hf_revision``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from huggingface_hub import HfApi
from packaging.version import Version
from packaging.version import parse as parse_version

# Hub repo that stores ONNX + tokenizer files versioned by Git tags.
REPO_ID = "alieissa/minilm-ui"


def resolve_weights_revision(package_version: str, *, token: Optional[str] = None) -> str:
    """Choose a Hugging Face Hub *revision* (branch name or tag name) for model weights.

    Resolution is deterministic given the same Hub tag list and *package_version*.

    **Algorithm**

    - Parse *package_version* with :func:`packaging.version.parse` (supports pre-releases
      and local segments per PEP 440).
    - Fetch all tag refs from the repo. Tags that are not valid single versions after
      stripping a leading ``v`` are ignored (e.g. ``latest``, ``v1-rc`` if unparsable).
    - If any remaining tag's version **equals** the package version, return that tag's
      **exact** ``ref.name`` (what the Hub API expects for ``revision=``).
    - Otherwise return the tag with the maximum version among those **strictly less than**
      the package version.
    - If there is no such tag, or the API call fails, return ``"main"``.

    **Robustness**

    - Network or auth errors when listing refs are swallowed so install/CI can still
      try ``main`` rather than crashing here; download may still fail later if ``main``
      is wrong or private.
    - Malformed tag names do not abort the scan; they are skipped.
    - Using ``ref.name`` preserves the real tag spelling (``v0.1.0`` vs ``0.1.0``).

    Args:
        package_version: Version string of the vizQA distribution (e.g. from
            ``importlib.metadata`` or ``pyproject.toml``).
        token: Optional Hugging Face token for private repos or higher rate limits.

    Returns:
        A revision string suitable for ``hf download --revision`` / ``snapshot_download``
        (``revision=``): a tag name, or ``"main"``.
    """
    target = parse_version(package_version)
    api = HfApi(token=token)

    try:
        tag_refs = api.list_repo_refs(repo_id=REPO_ID).tags
    except Exception:  # pylint: disable=broad-exception-caught
        # Listing refs can fail offline, on rate limits, or for auth issues.
        # Defer failure to the actual download if "main" is not usable.
        return "main"

    # (parsed Version, exact tag name as on Hub)
    parsed_tags: list[tuple[Version, str]] = []
    for ref in tag_refs:
        raw = ref.name.lstrip("v")
        try:
            parsed_tags.append((parse_version(raw), ref.name))
        except Exception:  # pylint: disable=broad-exception-caught
            continue

    # Prefer weights cut for this exact release.
    for v, name in parsed_tags:
        if v == target:
            return name

    # No vX.Y.Z for this bump: ship the newest weights older than this package version.
    best_name = "main"
    best_v = parse_version("0.0.0")
    for v, name in parsed_tags:
        if best_v < v < target:
            best_v = v
            best_name = name

    return best_name


def main() -> None:
    """CLI for CI: print one revision line to stdout; read token from ``HF_TOKEN``."""
    if len(sys.argv) != 2:
        print("usage: python -m vizQA.hf_revision <package-version>", file=sys.stderr)
        sys.exit(2)
    token = os.environ.get("HF_TOKEN") or None
    print(resolve_weights_revision(sys.argv[1], token=token))


if __name__ == "__main__":
    main()
