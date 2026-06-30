# Changelog

All notable changes to **vizQA** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-05-28

### Added

- New semantic `scroll to {elem}` support with full-range sweep handling, target centering, and edge-aware success when the page boundary prevents perfect centering.
- New semantic `wait for {elem}` support with 1s polling and configurable timeout/poll cadence.
- Runtime configuration knobs for wait-for timeout, wait-for poll interval, and scroll centering band.
- Config-backed reasoning language packs under `vizQA/reasoning/languages`, including the default English pack for action synonyms, semantic anchors, verification vocabulary, state/color/position terms, generic page-scope terms, and salience terms.
- A typed language-pack loader and canonical `Intent` model for parser, ranking, and runtime semantic reasoning.
- Extra semantic metadata in installed weights metadata, including language id, language schema version, provider id, and provider revision.

### Changed

- `scroll` commands now distinguish target-seeking scrolls from simple directional/page-boundary scrolls without breaking old timed `wait` behavior or `VERIFY` polling.
- Semantic parsing, ranking, verification, wait-for polling, scroll intent classification, and historical target resolution now use the canonical `Intent` object end to end instead of the older dict-shaped intent flow.
- MiniLM is now treated as a semantic provider implementation rather than the owner of product vocabulary; its action/state/color/position/negation anchors are sourced from the language pack instead of being hardcoded in the model adapter.
- Parser and planner/provider wiring now consistently use the semantic-provider boundary, removing leftover `parser.minilm` coupling and other legacy intent/model compatibility paths.
- Definition and planning failures are now reported as blocked test sessions instead of lane-level errors, so the failure stays attached to the relevant test case and the run continues.

## [0.3.1] - 2026-05-22

### Changed

- Shared pre-requisites are now reused once per viewport lane within a single `vizqa run` instead of being rerun for every top-level test that depends on them. Added `--no-cache` to explicitly force fresh pre-requisite reruns when needed.
- Cleaner failure messages for pre-requisite failures, and url connection issues.

## [0.3.0] - 2026-05-16

### Changed

- **Breaking: CLI reporting system overhauled** - Terminal reporter now uses an event-driven architecture with structured reporting events (`TopLevelTestStartedEvent`, `SessionStartedEvent`, `StepStartedEvent`, `StepFinishedEvent`, `SessionBlockedEvent`, `SessionFinishedEvent`, `RunFinishedEvent`). Custom reporters must implement the new `handle(event)` interface. The reporting store, layout composition, and terminal rendering have been completely redesigned for better multi-viewport support and step-level granularity.
- CLI terminal output now displays shared test flows with moving viewport cursors, improved dependency section rendering with progress bars, and better visual hierarchy during test execution.
- User-facing reporting and documentation now refer to `requires` setup flows as pre-requisites instead of dependencies, to reduce confusion with package/install dependencies.
- Fixed a bug where Type actions were not cleared before typing on MacOS.
- Adapted interactive cli mode with new reporting flow.

## [0.2.0] - 2026-04-24

### Added

- Community docs: Code of Conduct, security policy, changelog, and pull request template.
- Weight metadata tracking under `vizQA/weights`, version-aware `vizqa --version` output, and non-blocking warnings when installed model weights do not match the package's expected weights revision.
- Test dependencies via `requires`, including dependency graph resolution, circular-dependency and missing-dependency validation, dependency-first execution, dependency skip behavior on failure, artifact inheritance, browser-state reuse, and top-level-only result summaries.
- Dependency-system documentation and a general YAML test authoring guide under `docs/`.
- A local dependency auth lab fixture plus example dependency test flows covering password login, MFA, role elevation, checkout, returns, approvals, session resume, and simulated SSO.
- Custom viewport profiles and viewport-matrix execution for `vizqa run`, including built-in viewport presets, config-defined app viewports, raw `WIDTHxHEIGHT` CLI overrides, and per-viewport browser-state, artifact, and debug-log isolation.
- `${VAR}` environment-variable interpolation for YAML test files, including fail-fast validation for missing variables during test loading and dependency resolution.
- An embedded Python library API for attaching `vizQA` to an existing Playwright `Page`, including `attach`, `click`, `type`, `verify`, `run_step`, and `run_steps`, plus dedicated library documentation.

### Changed

- CLI runs now clear stale screenshots and browser-state caches for the requested tests and their dependency chain before execution, reducing artifact carryover between runs.
- CLI cleanup summaries are now shown only in verbose mode to keep default test output quieter.
- Dependency-related reporting is clearer during execution, including skip messaging and cleaner output for dependency-caused failures.
- Multi-viewport CLI reporting now renders a shared test flow with moving viewport cursors instead of duplicating the full step stream for each viewport lane.
- Library API usage is now artifact-light by default, keeping persistent screenshots only when `debug_dir` is explicitly provided.
- The small delay between interaction steps is now configurable through `VIZQA_STEP_DELAY_SECONDS` instead of being hardcoded.

## [0.1.0] - 2026-04-04

Initial published release line (alpha). See the [README](README.md) for features
and usage.

[0.4.0]: https://github.com/Spospider/vizQA/compare/v0.3.1...v0.4.0
[0.2.0]: https://github.com/Spospider/vizQA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Spospider/vizQA/releases/tag/v0.1.0
