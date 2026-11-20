import pytest
import yaml

eval_utils = pytest.importorskip("hirad.eval.eval_utils")


@pytest.fixture
def eval_config(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({
        "inference_output_dir": "/path/to/generation/results/directory",
        "times_ranges": [["20210601-0000", "20210601-0200", 1]],
        "wet_threshold": 0.1,
    }))
    return str(path)


class TestParseEvalCli:
    def test_config_as_written(self, eval_config):
        cfg = eval_utils.parse_eval_cli(argv=["--config-name", eval_config])
        assert cfg["inference_output_dir"] == "/path/to/generation/results/directory"
        assert cfg["wet_threshold"] == 0.1

    def test_inference_output_dir_override(self, eval_config):
        cfg = eval_utils.parse_eval_cli(argv=["--config-name", eval_config, "--inference-output-dir", "/runs/gen1"])
        assert cfg["inference_output_dir"] == "/runs/gen1"
        assert cfg["times_ranges"] == [["20210601-0000", "20210601-0200", 1]]

    def test_times_override_replaces_ranges(self, eval_config):
        cfg = eval_utils.parse_eval_cli(
            allow_times=True,
            argv=["--config-name", eval_config, "--times", "20210601-0100", "--inference-output-dir", "/runs/gen1"],
        )
        assert cfg["times"] == ["20210601-0100"]
        assert "times_ranges" not in cfg
        assert cfg["inference_output_dir"] == "/runs/gen1"

    def test_times_rejected_unless_allowed(self, eval_config):
        with pytest.raises(SystemExit):
            eval_utils.parse_eval_cli(argv=["--config-name", eval_config, "--times", "20210601-0100"])
