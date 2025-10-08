#!/usr/bin/env python3
"""
Test that stub files are complete and properly expose all classes/functions
"""

import ast
import os
from pathlib import Path


def parse_stub_exports(stub_path):
    """
    Parse a .pyi stub file and extract all class/function names from __all__
    """
    with open(stub_path, 'r') as f:
        tree = ast.parse(f.read())

    # Find __all__ assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == '__all__':
                if isinstance(node.value, ast.List):
                    result = set()
                    for elt in node.value.elts:
                        # Handle both ast.Str (Python 3.7) and ast.Constant (Python 3.8+)
                        if hasattr(elt, 's'):
                            result.add(elt.s)
                        elif hasattr(elt, 'value'):
                            result.add(elt.value)
                    return result
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List):
                        result = set()
                        for elt in node.value.elts:
                            if hasattr(elt, 's'):
                                result.add(elt.s)
                            elif hasattr(elt, 'value'):
                                result.add(elt.value)
                        return result
    return set()


def parse_stub_classes_and_functions(stub_path):
    """
    Parse a .pyi stub file and extract all top-level class and function names
    """
    with open(stub_path, 'r') as f:
        tree = ast.parse(f.read())

    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
    return names


def parse_stub_imports(stub_path):
    """
    Parse a .pyi stub file and extract all imports from a specific module
    """
    with open(stub_path, 'r') as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module and 'factorama._factorama' in node.module:
                for alias in node.names:
                    imports.add(alias.name)
    return imports


def test_stub_completeness():
    """
    Test that all classes/functions in _factorama.pyi are properly exported in __init__.pyi
    """
    # Find stub file paths
    repo_root = Path(__file__).parent.parent.parent
    factorama_stub = repo_root / 'python_bindings' / 'python' / 'factorama' / '_factorama.pyi'
    init_stub = repo_root / 'python_bindings' / 'python' / 'factorama' / '__init__.pyi'

    assert factorama_stub.exists(), f"_factorama.pyi not found at {factorama_stub}"
    assert init_stub.exists(), f"__init__.pyi not found at {init_stub}"

    # Get all classes and functions from _factorama.pyi
    factorama_exports = parse_stub_exports(factorama_stub)
    factorama_classes_funcs = parse_stub_classes_and_functions(factorama_stub)

    # Get imports and __all__ from __init__.pyi
    init_imports = parse_stub_imports(init_stub)
    init_all = parse_stub_exports(init_stub)

    print(f"\n_factorama.pyi exports ({len(factorama_exports)} items):")
    print(f"  {sorted(factorama_exports)}")

    print(f"\n__init__.pyi imports from _factorama ({len(init_imports)} items):")
    print(f"  {sorted(init_imports)}")

    print(f"\n__init__.pyi __all__ ({len(init_all)} items):")
    print(f"  {sorted(init_all)}")

    # Check 1: All classes/functions from _factorama.pyi should be in its __all__
    missing_in_factorama_all = factorama_classes_funcs - factorama_exports
    if missing_in_factorama_all:
        print(f"\n⚠️  WARNING: Classes/functions missing from _factorama.pyi __all__:")
        print(f"  {sorted(missing_in_factorama_all)}")

    # Check 2: All items from _factorama.pyi __all__ should be imported in __init__.pyi
    missing_imports = factorama_exports - init_imports
    if missing_imports:
        print(f"\n❌ ERROR: Classes/functions in _factorama.pyi but NOT imported in __init__.pyi:")
        for name in sorted(missing_imports):
            print(f"  - {name}")
        print(f"\nAdd these lines to __init__.pyi:")
        for name in sorted(missing_imports):
            print(f"from factorama._factorama import {name}")
        assert False, f"Missing imports in __init__.pyi: {missing_imports}"

    # Check 3: All imports should be in __init__.pyi __all__ (excluding helper items)
    # Filter out things that shouldn't be in __all__ (like PlotFactorGraph which is defined in __init__.py)
    expected_in_all = init_imports
    missing_in_all = expected_in_all - init_all
    if missing_in_all:
        print(f"\n❌ ERROR: Imports missing from __init__.pyi __all__:")
        for name in sorted(missing_in_all):
            print(f"  - {name}")
        print(f"\nAdd these to __init__.pyi __all__ list:")
        print(f"  {sorted(missing_in_all)}")
        assert False, f"Missing items in __init__.pyi __all__: {missing_in_all}"

    print("\n✅ All stub files are complete and consistent!")


if __name__ == "__main__":
    test_stub_completeness()
