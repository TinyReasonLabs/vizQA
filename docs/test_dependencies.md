# Test Dependencies

vizQA lets one test depend on another with `requires`. This is useful for flows like:
- login -> MFA
- login -> checkout -> returns
- login -> access request + role elevation -> manager approval

## Simple Usage

Define the prerequisite test:

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

Then declare a dependent test:

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

## Notes For Writing Tests

- `requires` entries reference file stems, not full filenames.
- Keep each test valid on its own. A dependency should represent a meaningful checkpoint.
- If a test uses `{artifacts}`, define them in that file even if a dependency also defines them.
- Use dependencies for stateful setup, not for grouping unrelated assertions.

## Technical Details

### Resolution

- Dependencies are resolved from the same directory as the test file.
- vizQA builds a dependency graph and runs dependencies in topological order.
- Circular dependencies and missing dependency names raise a `TestDefinitionError`.

### Execution Model

- Dependency tests are executed before the requested test.
- If any dependency fails, the requested test is skipped.
- Dependency sessions are reported in the output, but summary pass/fail counts only include top-level requested tests.

### Artifacts

- Artifacts from dependencies are loaded and inherited in dependency order.
- Local artifacts override inherited artifacts with the same name.
- In practice, explicit local artifacts are recommended for readability.

### Browser State

- After a successful dependency, vizQA caches browser state from:
  - `localStorage`
  - `sessionStorage`
  - cookies
- Before the dependent test starts, vizQA restores the latest dependency state and reloads the page so storage-driven UI can rehydrate.

### Limits

- Only persisted browser state is transferred automatically. If an app keeps important UI state only in live memory, that state should also be reflected in storage or cookies.
- Dependencies are file-based; there is no separate suite manifest.
