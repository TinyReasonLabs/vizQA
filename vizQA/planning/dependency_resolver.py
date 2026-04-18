"""
Dependency resolver for test cases with support for circular dependency detection.
"""

from pathlib import Path
from typing import Dict, List, Set

import yaml

from vizQA.app.exceptions import TestDefinitionError
from vizQA.utils import LineLoader


# pylint: disable=too-few-public-methods
class DependencyResolver:
    """Resolves test dependencies and detects circular references."""

    def __init__(self, test_dir: Path):
        """
        Initialize the resolver with a test directory.

        :param test_dir: Path to the directory containing test files
        """
        self.test_dir = Path(test_dir)
        self._test_cache: Dict[str, Path] = {}
        self._load_test_files()

    def _load_test_files(self) -> None:
        """Load all test files in the test directory and cache their paths."""
        for test_file in self.test_dir.glob("*.yaml"):
            self._test_cache[test_file.stem] = test_file
        for test_file in self.test_dir.glob("*.yml"):
            self._test_cache[test_file.stem] = test_file

    def _find_test_file(self, test_name: str) -> Path:
        """
        Find a test file by name (stem).

        :param test_name: Name of the test (without .yaml/.yml extension)
        :return: Path to the test file
        :raises TestDefinitionError: If test not found
        """
        if test_name in self._test_cache:
            return self._test_cache[test_name]

        raise TestDefinitionError(
            f"Required test '{test_name}' not found",
            internal_detail=f"Searched in {self.test_dir}. Available tests: {', '.join(self._test_cache.keys())}",
        )

    def _load_test_data(self, test_path: Path) -> dict:
        """Load YAML test data from a file."""
        try:
            return yaml.load(test_path.read_text(encoding="utf-8"), Loader=LineLoader)
        except Exception as err:
            raise TestDefinitionError(
                f"Failed to load test file {test_path.name}",
                internal_detail=str(err),
            ) from err

    def _detect_circular_dependencies(
        self, test_name: str, visited: Set[str], rec_stack: Set[str], graph: Dict[str, List[str]]
    ) -> None:
        """
        Detect circular dependencies using DFS.

        :param test_name: Current test being visited
        :param visited: Set of all visited tests
        :param rec_stack: Current recursion stack
        :param graph: Dependency graph mapping test names to their required tests
        :raises TestDefinitionError: If circular dependency detected
        """
        visited.add(test_name)
        rec_stack.add(test_name)

        for dependency in graph.get(test_name, []):
            if dependency not in visited:
                self._detect_circular_dependencies(dependency, visited, rec_stack, graph)
            elif dependency in rec_stack:
                raise TestDefinitionError(
                    f"Circular dependency detected: {test_name} → {dependency}",
                    internal_detail=f"Test {test_name} depends on {dependency}, \
                    which eventually depends on {test_name}.",
                )

        rec_stack.remove(test_name)

    def _build_dependency_graph(self, test_name: str, graph: Dict[str, List[str]]) -> None:
        """
        Build a dependency graph by recursively loading test definitions.

        :param test_name: Test to start from
        :param graph: Graph to populate
        """
        if test_name in graph:
            return  # Already processed

        test_path = self._find_test_file(test_name)
        test_data = self._load_test_data(test_path)

        requires = test_data.get("requires", [])
        graph[test_name] = requires if isinstance(requires, list) else []

        for dependency in graph[test_name]:
            self._build_dependency_graph(dependency, graph)

    def resolve(self, test_path: Path) -> List[Path]:
        """
        Resolve all dependencies for a test in topological order.

        :param test_path: Path to the test file
        :return: List of test paths in order (dependencies first, then the main test)
        :raises TestDefinitionError: If circular dependency or missing test detected
        """
        test_name = test_path.stem

        # Build complete dependency graph
        graph: Dict[str, List[str]] = {}
        self._build_dependency_graph(test_name, graph)

        # Detect circular dependencies
        visited: Set[str] = set()
        for node in graph:
            if node not in visited:
                self._detect_circular_dependencies(node, visited, set(), graph)

        # Topological sort using DFS
        visited.clear()
        sorted_tests: List[str] = []

        def topological_sort(node: str) -> None:
            visited.add(node)
            for dependency in graph.get(node, []):
                if dependency not in visited:
                    topological_sort(dependency)
            sorted_tests.append(node)

        topological_sort(test_name)

        # Convert test names to paths (exclude the main test from results)
        result = []
        for name in sorted_tests[:-1]:  # Exclude the last one (main test)
            result.append(self._find_test_file(name))

        return result
