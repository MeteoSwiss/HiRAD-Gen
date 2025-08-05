"""
Environment Introspection Module
================================

This module provides functionality to introspect the Python environment, listing all non-standard
library modules along with their versions and Git information if they are part of a Git repository.
It helps in understanding the environment setup by detailing the modules in use, their versions, and
relevant Git metadata.
"""

import platform
import os
import sys
from types import ModuleType
from typing import Any, Optional
from git import Repo, InvalidGitRepositoryError


def get_module_version(module: ModuleType) -> Optional[str]:
    """
    Retrieve the version of a module if available.

    This function attempts to get the version of a module by accessing its ``__version__``
    attribute. It checks if the version is a string to ensure correctness.

    :param module: The module whose version is to be retrieved.
    :type module: ModuleType
    :return: The version string if available and valid, otherwise None.
    :rtype: Optional[str]
    """
    version = getattr(module, "__version__", None)
    if isinstance(version, str):
        return version
    return None


def get_git_info(path: str) -> Optional[dict[str, Any]]:
    """
    Collect basic Git metadata for a given repository path.

    This function checks if the given path is part of a Git repository and collects metadata such as
    the commit SHA, modified files, untracked files, and remote URLs.

    :param path: The path to check for Git repository metadata.
    :type path: str
    :return: A dictionary containing Git metadata if the path is a Git repository, otherwise None.
    :rtype: Optional[Dict[str, Any]]
    """
    try:
        repo = Repo(path, search_parent_directories=True)
        diff = ""
        for diff_item in repo.index.diff(None, create_patch=True):
            a_path = diff_item.a_blob.abspath if diff_item.a_blob else ""
            b_path = diff_item.b_blob.abspath if diff_item.b_blob else ""
            diff_content = diff_item.diff
            if isinstance(diff_content, bytes):
                diff_content = diff_content.decode("utf-8")
            elif diff_content is None:
                diff_content = ""
            diff += f"--- a{a_path}\n+++ b{b_path}\n{diff_content}\n\n"
        git_info = {
            "sha1": repo.head.commit.hexsha,
            "diff": diff,
            "untracked_files": sorted(repo.untracked_files),
            "remotes": [r.url for r in repo.remotes],
        }
        return git_info
    except InvalidGitRepositoryError:
        return None


def get_module_git_info(module: ModuleType) -> Optional[dict[str, Any]]:
    """
    Get Git information for a module if its directory is a Git repository.

    This function determines the directory of the module and checks if it is part of a Git
    repository. If it is, it collects and returns the Git metadata.

    :param module: The module to check for Git information.
    :type module: ModuleType
    :return: A dictionary containing Git metadata if the module is in a Git repository, otherwise
    None.
    :rtype: Optional[Dict[str, Any]]
    """
    module_path = getattr(module, "__file__", None)
    if module_path is None or not os.path.isabs(module_path):
        return None
    module_dir = os.path.dirname(module_path)
    return get_git_info(module_dir)


def get_env_info(flatten: bool = True, exclude_prefixes: list[str] = None) -> tuple[dict[str, dict[str, Any]], str]:
    """
    List all non-standard library modules with their versions and Git information if available.

    This function iterates over all loaded modules in the Python environment, filtering out
    built-in modules. It collects version and Git information for each remaining module.

    :return: A tuple containing two elements:
        - A dictionary mapping module names to their metadata
        - A string containing concatenated Git diffs from all modules
    :rtype: tuple[dict[str, dict[str, Any]], str]
    :rtype: Dict[str, Dict[str, Any]]
    """
    env_info = {}
    diffs: list[str] = []

    exclude_prefixes = exclude_prefixes or []

    for name, module in sys.modules.copy().items():
        if name in sys.builtin_module_names or name.endswith(('.version','._version')):
            continue

        if any(name == prefix or name.startswith(prefix + ".") for prefix in exclude_prefixes):
            continue

        version = get_module_version(module)
        git_info = get_module_git_info(module)
        if version is None and git_info is None:
            continue

        module_info: dict[str, Any] = {"version": version}
        if git_info is not None:
            module_diff = git_info.pop("diff")
            if module_diff and module_diff not in diffs:
                diffs.append(module_diff)
            module_info["git"] = git_info

        env_info[name] = module_info

    env_info["python"] = {"version": platform.python_version()}

    diffs_str = "\n".join(diffs)

    if flatten:
        return flatten_dict(env_info), diffs_str

    return env_info, diffs_str


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested dictionary.

    This function recursively traverses a nested dictionary and flattens it into a single-level
    dictionary with keys formed by concatenating the nested keys using a separator.

    :param d: The dictionary to flatten.
    :type d: Dict[str, Any]
    :param parent_key: The base key to use for concatenation.
    :type parent_key: str
    :param sep: The separator to use for concatenating keys.
    :type sep: str
    :return: A flattened dictionary.
    :rtype: Dict[str, Any]
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
