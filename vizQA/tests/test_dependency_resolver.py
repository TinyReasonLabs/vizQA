"""
Tests for the dependency resolver module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from vizQA.app.exceptions import TestDefinitionError
from vizQA.planning import DependencyResolver


class TestDependencyResolver:
    """Test cases for the DependencyResolver class."""

    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_test_file(self, test_dir: Path, name: str, requires: list = None):
        """Helper to create a test file."""
        test_file = test_dir / f"{name}.yaml"
        data = {
            "name": name.replace("_", " ").title(),
            "url": "http://example.com",
            "steps": [{"action": f"Do something for {name}", "expect": "Something happens"}],
        }
        if requires:
            data["requires"] = requires

        with open(test_file, "w") as f:
            yaml.dump(data, f)

        return test_file

    def test_no_dependencies(self, temp_test_dir):
        """Test a test file with no dependencies."""
        test_file = self.create_test_file(temp_test_dir, "standalone")
        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(test_file)
        assert deps == []

    def test_single_dependency(self, temp_test_dir):
        """Test resolving a single dependency."""
        self.create_test_file(temp_test_dir, "login")
        self.create_test_file(temp_test_dir, "checkout", ["login"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "checkout.yaml")

        assert len(deps) == 1
        assert deps[0].stem == "login"

    def test_transitive_dependencies(self, temp_test_dir):
        """Test resolving transitive dependencies."""
        self.create_test_file(temp_test_dir, "auth")
        self.create_test_file(temp_test_dir, "login", ["auth"])
        self.create_test_file(temp_test_dir, "checkout", ["login"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "checkout.yaml")

        assert len(deps) == 2
        assert deps[0].stem == "auth"
        assert deps[1].stem == "login"

    def test_circular_dependency_direct(self, temp_test_dir):
        """Test detection of direct circular dependencies."""
        self.create_test_file(temp_test_dir, "a", ["b"])
        self.create_test_file(temp_test_dir, "b", ["a"])

        resolver = DependencyResolver(temp_test_dir)
        with pytest.raises(TestDefinitionError) as exc_info:
            resolver.resolve(temp_test_dir / "a.yaml")

        assert "Circular dependency detected" in str(exc_info.value)

    def test_circular_dependency_indirect(self, temp_test_dir):
        """Test detection of indirect circular dependencies."""
        self.create_test_file(temp_test_dir, "a", ["b"])
        self.create_test_file(temp_test_dir, "b", ["c"])
        self.create_test_file(temp_test_dir, "c", ["a"])

        resolver = DependencyResolver(temp_test_dir)
        with pytest.raises(TestDefinitionError) as exc_info:
            resolver.resolve(temp_test_dir / "a.yaml")

        assert "Circular dependency detected" in str(exc_info.value)

    def test_missing_dependency(self, temp_test_dir):
        """Test error when a required test is not found."""
        self.create_test_file(temp_test_dir, "checkout", ["nonexistent_login"])

        resolver = DependencyResolver(temp_test_dir)
        with pytest.raises(TestDefinitionError) as exc_info:
            resolver.resolve(temp_test_dir / "checkout.yaml")

        assert "not found" in str(exc_info.value)

    def test_multiple_dependencies(self, temp_test_dir):
        """Test resolving multiple independent dependencies."""
        self.create_test_file(temp_test_dir, "auth")
        self.create_test_file(temp_test_dir, "setup")
        self.create_test_file(temp_test_dir, "checkout", ["auth", "setup"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "checkout.yaml")

        assert len(deps) == 2
        dep_stems = {d.stem for d in deps}
        assert dep_stems == {"auth", "setup"}

    def test_fork_pattern(self, temp_test_dir):
        """Test the fork pattern - multiple tests depending on the same test."""
        self.create_test_file(temp_test_dir, "login")
        self.create_test_file(temp_test_dir, "checkout", ["login"])
        self.create_test_file(temp_test_dir, "returns", ["login"])

        resolver = DependencyResolver(temp_test_dir)

        # Checkout should depend on login
        checkout_deps = resolver.resolve(temp_test_dir / "checkout.yaml")
        assert len(checkout_deps) == 1
        assert checkout_deps[0].stem == "login"

        # Returns should also depend on login
        returns_deps = resolver.resolve(temp_test_dir / "returns.yaml")
        assert len(returns_deps) == 1
        assert returns_deps[0].stem == "login"

    def test_branching_graph_preserves_dependency_order(self, temp_test_dir):
        """Test a graph where one shared dependency fans out into two branches."""
        self.create_test_file(temp_test_dir, "seed")
        self.create_test_file(temp_test_dir, "password_login", ["seed"])
        self.create_test_file(temp_test_dir, "mfa_login", ["password_login"])
        self.create_test_file(temp_test_dir, "access_request", ["mfa_login"])
        self.create_test_file(temp_test_dir, "role_elevation", ["mfa_login"])
        self.create_test_file(temp_test_dir, "manager_approval", ["role_elevation", "access_request"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "manager_approval.yaml")

        assert [dep.stem for dep in deps] == [
            "seed",
            "password_login",
            "mfa_login",
            "role_elevation",
            "access_request",
        ]

    def test_four_level_dependency_chain(self, temp_test_dir):
        """Test a four-level chain used by checkout -> returns flows."""
        self.create_test_file(temp_test_dir, "seed")
        self.create_test_file(temp_test_dir, "password_login", ["seed"])
        self.create_test_file(temp_test_dir, "mfa_login", ["password_login"])
        self.create_test_file(temp_test_dir, "checkout", ["mfa_login"])
        self.create_test_file(temp_test_dir, "returns", ["checkout"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "returns.yaml")

        assert [dep.stem for dep in deps] == [
            "seed",
            "password_login",
            "mfa_login",
            "checkout",
        ]

    def test_shared_dependency_only_appears_once_per_graph(self, temp_test_dir):
        """Test that a shared prerequisite is not duplicated within one resolved graph."""
        self.create_test_file(temp_test_dir, "seed")
        self.create_test_file(temp_test_dir, "password_login", ["seed"])
        self.create_test_file(temp_test_dir, "mfa_login", ["password_login"])
        self.create_test_file(temp_test_dir, "checkout", ["mfa_login"])
        self.create_test_file(temp_test_dir, "access_request", ["mfa_login"])
        self.create_test_file(temp_test_dir, "dual_branch", ["checkout", "access_request"])

        resolver = DependencyResolver(temp_test_dir)
        deps = resolver.resolve(temp_test_dir / "dual_branch.yaml")

        stems = [dep.stem for dep in deps]
        assert stems.count("mfa_login") == 1
        assert stems.count("password_login") == 1
        assert stems.count("seed") == 1
