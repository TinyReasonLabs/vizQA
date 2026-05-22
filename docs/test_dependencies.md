# Test Pre-requisites

This guide covers the user-facing pre-requisite flow powered by vizQA's internal dependency system.

vizQA lets one test depend on another with `requires`. This is useful for flows like:
- login -> MFA
- login -> checkout -> returns
- login -> access request + role elevation -> manager approval

## Simple Usage

Define the pre-requisite test:

```yaml
# login.yaml
name: "Login"
url: "https://example.com/login"

artifacts:
  username: "analyst.user"
  password: "AnalystPass!23"

steps:
  - action: "Type {username} into the username field"
  - action: "Type {password} into the password field"
  - action: "Click the 'Sign in' button"
    expect: "The dashboard should load"
```

Then declare a test that depends on it:

```yaml
# returns.yaml
name: "Returns"
url: "https://example.com/app"

requires:
  - login

steps:
  - action: "Click the 'Orders' navigation button"
  - action: "Click the 'Request Return' button"
    expect: "A return request form should appear"
```

Run the dependent test normally:

```bash
vizqa tests/returns.yaml
```

vizQA will:
1. Resolve `login`
2. Run `login` first
3. Reuse its browser state
4. Start `returns`

When multiple requested tests share the same pre-requisite in a single run, vizQA now reuses the first successful execution within that viewport lane by default. Use `--no-cache` to force every requested test to rerun its full pre-requisite chain.

## Notes For Writing Tests

- `requires` entries reference file stems, not full filenames.
- Keep each test valid on its own. A pre-requisite should represent a meaningful checkpoint.
- If a test uses `{artifacts}`, define them in that file even if a pre-requisite also defines them.
- Use pre-requisites for stateful setup, not for grouping unrelated assertions.

## Technical Details

### Resolution

- Pre-requisites are resolved from the same directory as the test file.
- vizQA builds an internal dependency graph and runs pre-requisites in topological order.
- Circular references and missing pre-requisite names raise a `TestDefinitionError`.

### Execution Model

- Pre-requisite tests are executed before the requested test.
- Within a single run, shared pre-requisites are executed once per viewport lane by default and then reused by later dependents.
- If any pre-requisite fails, the requested test is skipped.
- Pre-requisite sessions are reported in the output, but summary pass/fail counts only include top-level requested tests.
- Pass `--no-cache` if you want to explicitly rerun pre-requisites for every requested test.

### Artifacts

- Artifacts from pre-requisites are loaded and inherited in pre-requisite order.
- Local artifacts override inherited artifacts with the same name.
- In practice, explicit local artifacts are recommended for readability.

### Browser State

- After a successful pre-requisite, vizQA caches browser state from:
  - `localStorage`
  - `sessionStorage`
  - cookies
- Before the dependent test starts, vizQA restores the latest pre-requisite state and reloads the page so storage-driven UI can rehydrate.

### Limits

- Only persisted browser state is transferred automatically. If an app keeps important UI state only in live memory, that state should also be reflected in storage or cookies.
- Pre-requisites are file-based; there is no separate suite manifest.
