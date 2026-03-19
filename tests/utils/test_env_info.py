import os
from types import ModuleType
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from hirad.utils.env_info import (
    flatten_dict,
    get_env_info,
    get_git_info,
    get_module_git_info,
    get_module_version,
)


############################################################################
#                          get_module_version                              #
############################################################################


class TestGetModuleVersion:
    """Tests for get_module_version."""

    def test_returns_version_string(self):
        module = ModuleType("fake_mod")
        module.__version__ = "1.2.3"
        assert get_module_version(module) == "1.2.3"

    def test_returns_none_when_no_version_attr(self):
        module = ModuleType("no_ver")
        assert get_module_version(module) is None

    def test_returns_none_when_version_is_not_string(self):
        module = ModuleType("bad_ver")
        module.__version__ = (1, 2, 3)
        assert get_module_version(module) is None

    def test_returns_none_when_version_is_none(self):
        module = ModuleType("none_ver")
        module.__version__ = None
        assert get_module_version(module) is None

    def test_returns_none_when_version_is_int(self):
        module = ModuleType("int_ver")
        module.__version__ = 42
        assert get_module_version(module) is None

    def test_accepts_semver_string(self):
        module = ModuleType("semver")
        module.__version__ = "0.0.1-alpha"
        assert get_module_version(module) == "0.0.1-alpha"


############################################################################
#                            get_git_info                                  #
############################################################################


class TestGetGitInfo:
    """Tests for get_git_info."""

    def test_returns_none_for_non_git_path(self, tmp_path):
        result = get_git_info(str(tmp_path))
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        mock_remote = MagicMock()
        mock_remote.url = "https://github.com/user/repo.git"

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = []
        mock_repo.remotes = [mock_remote]

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert result is not None
        assert result["sha1"] == "abc123"
        assert result["diff"] == ""
        assert result["untracked_files"] == []
        assert result["remotes"] == ["https://github.com/user/repo.git"]

    def test_untracked_files_are_sorted(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "def456"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = ["c.py", "a.py", "b.py"]
        mock_repo.remotes = []

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert result["untracked_files"] == ["a.py", "b.py", "c.py"]

    def test_diff_content_bytes_decoded(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "aaa"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        diff_item = MagicMock()
        diff_item.a_blob.abspath = "/a/file.py"
        diff_item.b_blob.abspath = "/b/file.py"
        diff_item.diff = b"+new line"

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = [diff_item]
        mock_repo.untracked_files = []
        mock_repo.remotes = []

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert "+new line" in result["diff"]

    def test_diff_content_string_passthrough(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "bbb"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        diff_item = MagicMock()
        diff_item.a_blob.abspath = "/a/file.py"
        diff_item.b_blob.abspath = "/b/file.py"
        diff_item.diff = "+already a string"

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = [diff_item]
        mock_repo.untracked_files = []
        mock_repo.remotes = []

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert "+already a string" in result["diff"]

    def test_diff_content_none_treated_as_empty(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "ccc"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        diff_item = MagicMock()
        diff_item.a_blob.abspath = "/a/file.py"
        diff_item.b_blob.abspath = "/b/file.py"
        diff_item.diff = None

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = [diff_item]
        mock_repo.untracked_files = []
        mock_repo.remotes = []

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        # The diff should contain the header lines but no diff content
        assert "--- a/a/file.py" in result["diff"]
        assert "+++ b/b/file.py" in result["diff"]

    def test_diff_multiple_diff_items_concatenated(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "eee"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        diff_item1 = MagicMock()
        diff_item1.a_blob.abspath = "/a/file1.py"
        diff_item1.b_blob.abspath = "/b/file1.py"
        diff_item1.diff = "+line in file1"

        diff_item2 = MagicMock()
        diff_item2.a_blob.abspath = "/a/file2.py"
        diff_item2.b_blob.abspath = "/b/file2.py"
        diff_item2.diff = "+line in file2"

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = [diff_item1, diff_item2]
        mock_repo.untracked_files = []
        mock_repo.remotes = []

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert "+line in file1" in result["diff"]
        assert "+line in file2" in result["diff"]
        assert result["diff"].count("--- a/") == 2
        assert result["diff"].count("+++ b/") == 2
        assert diff_item1.a_blob.abspath in result["diff"]
        assert diff_item1.b_blob.abspath in result["diff"]
        assert diff_item2.a_blob.abspath in result["diff"]
        assert diff_item2.b_blob.abspath in result["diff"]

    def test_multiple_remotes(self):
        mock_commit = MagicMock()
        mock_commit.hexsha = "ddd"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        remote1 = MagicMock()
        remote1.url = "https://github.com/user/repo1.git"
        remote2 = MagicMock()
        remote2.url = "git@github.com:user/repo2.git"

        mock_repo = MagicMock()
        mock_repo.head = mock_head
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = []
        mock_repo.remotes = [remote1, remote2]

        with patch("hirad.utils.env_info.Repo", return_value=mock_repo):
            result = get_git_info("/some/path")

        assert len(result["remotes"]) == 2


############################################################################
#                         get_module_git_info                              #
############################################################################


class TestGetModuleGitInfo:
    """Tests for get_module_git_info."""

    def test_returns_none_when_no_file_attr(self):
        module = ModuleType("no_file")
        # ModuleType does not set __file__ by default
        assert get_module_git_info(module) is None

    def test_returns_none_for_relative_path(self):
        module = ModuleType("rel_path")
        module.__file__ = "relative/path.py"
        assert get_module_git_info(module) is None

    def test_delegates_to_get_git_info(self, tmp_path):
        module = ModuleType("abs_path")
        module.__file__ = str(tmp_path / "pkg" / "mod.py")

        fake_git_info = {"sha1": "abc", "diff": "", "untracked_files": [], "remotes": []}
        with patch("hirad.utils.env_info.get_git_info", return_value=fake_git_info) as mock_fn:
            result = get_module_git_info(module)

        mock_fn.assert_called_once_with(str(tmp_path / "pkg"))
        assert result == fake_git_info

    def test_returns_none_when_get_git_info_returns_none(self, tmp_path):
        module = ModuleType("no_git")
        module.__file__ = str(tmp_path / "mod.py")

        with patch("hirad.utils.env_info.get_git_info", return_value=None):
            assert get_module_git_info(module) is None


############################################################################
#                            flatten_dict                                  #
############################################################################


class TestFlattenDict:
    """Tests for flatten_dict."""

    def test_already_flat(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_one_level_nesting(self):
        d = {"a": {"x": 1, "y": 2}, "b": 3}
        assert flatten_dict(d) == {"a.x": 1, "a.y": 2, "b": 3}

    def test_two_level_nesting(self):
        d = {"a": {"b": {"c": 42}}}
        assert flatten_dict(d) == {"a.b.c": 42}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d, sep="/") == {"a/b": 1}

    def test_custom_parent_key(self):
        d = {"x": 1}
        assert flatten_dict(d, parent_key="root") == {"root.x": 1}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_mixed_nested_and_flat(self):
        d = {"a": 1, "b": {"c": 2}, "d": {"e": {"f": 3}}}
        expected = {"a": 1, "b.c": 2, "d.e.f": 3}
        assert flatten_dict(d) == expected

    def test_preserves_non_dict_values(self):
        d = {"a": [1, 2], "b": {"c": "text"}, "d": None}
        expected = {"a": [1, 2], "b.c": "text", "d": None}
        assert flatten_dict(d) == expected


############################################################################
#                            get_env_info                                  #
############################################################################


class TestGetEnvInfo:
    """Tests for get_env_info."""

    def _make_module(self, name, version=None, file_path=None):
        """Create a fake module with optional version and __file__."""
        mod = ModuleType(name)
        if version is not None:
            mod.__version__ = version
        if file_path is not None:
            mod.__file__ = file_path
        return mod

    def test_returns_tuple_of_two(self):
        with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
            result = get_env_info()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_always_includes_python_version(self):
        with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
            info, _ = get_env_info(flatten=False)
        assert "python" in info
        assert "version" in info["python"]

    def test_flatten_true_flattens_output(self):
        with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
            info, _ = get_env_info(flatten=True)
        assert "python.version" in info

    def test_flatten_false_keeps_nested(self):
        with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
            info, _ = get_env_info(flatten=False)
        assert isinstance(info.get("python"), dict)

    def test_excludes_builtin_modules(self):
        import sys

        with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
            info, _ = get_env_info(flatten=False)
        for name in sys.builtin_module_names:
            assert name not in info

    def test_exclude_prefixes_filters_modules(self):
        fake_mod = self._make_module("fakepkg_test", version="1.0.0")
        with patch.dict("sys.modules", {"fakepkg_test": fake_mod}):
            with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
                info, _ = get_env_info(flatten=False, exclude_prefixes=["fakepkg_test"])
        assert "fakepkg_test" not in info

    def test_exclude_prefixes_filters_submodules(self):
        fake_sub = self._make_module("fakepkg_test.sub", version="2.0.0")
        with patch.dict("sys.modules", {"fakepkg_test.sub": fake_sub}):
            with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
                info, _ = get_env_info(flatten=False, exclude_prefixes=["fakepkg_test"])
        assert "fakepkg_test.sub" not in info

    def test_module_with_version_included(self):
        fake_mod = self._make_module("mypkg_env_test", version="3.1.4")
        with patch.dict("sys.modules", {"mypkg_env_test": fake_mod}):
            with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
                info, _ = get_env_info(flatten=False)
        assert "mypkg_env_test" in info
        assert info["mypkg_env_test"]["version"] == "3.1.4"

    def test_module_with_git_info_included(self):
        fake_mod = self._make_module("gitpkg_test", version="0.1.0")
        git_info = {
            "sha1": "abc123",
            "diff": "",
            "untracked_files": [],
            "remotes": ["https://example.com/repo.git"],
        }
        with patch.dict("sys.modules", {"gitpkg_test": fake_mod}):
            with patch(
                "hirad.utils.env_info.get_module_git_info",
                side_effect=lambda m: git_info.copy(),
            ):
                info, _ = get_env_info(flatten=False)
        assert "gitpkg_test" in info
        assert "git" in info["gitpkg_test"]
        assert info["gitpkg_test"]["git"]["sha1"] == "abc123"
        assert "diff" not in info["gitpkg_test"]["git"]

    def test_diffs_collected_in_second_element(self):
        fake_mod = self._make_module("diffpkg_test", version="1.0.0")
        git_info = {
            "sha1": "aaa",
            "diff": "+added line\n",
            "untracked_files": [],
            "remotes": [],
        }
        with patch.dict("sys.modules", {"diffpkg_test": fake_mod}):
            with patch(
                "hirad.utils.env_info.get_module_git_info",
                side_effect=lambda m: git_info.copy(),
            ):
                _, diffs_str = get_env_info(flatten=False)
        assert "+added line" in diffs_str

    def test_modules_without_version_and_git_excluded(self):
        fake_mod = self._make_module("bare_mod_test")
        with patch.dict("sys.modules", {"bare_mod_test": fake_mod}):
            with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
                info, _ = get_env_info(flatten=False)
        assert "bare_mod_test" not in info

    def test_version_suffix_modules_skipped(self):
        """Modules ending with .version or ._version are skipped."""
        fake_ver = self._make_module("somepkg.version", version="1.0")
        fake_uver = self._make_module("somepkg._version", version="1.0")
        with patch.dict(
            "sys.modules",
            {"somepkg.version": fake_ver, "somepkg._version": fake_uver},
        ):
            with patch("hirad.utils.env_info.get_module_git_info", return_value=None):
                info, _ = get_env_info(flatten=False)
        assert "somepkg.version" not in info
        assert "somepkg._version" not in info
