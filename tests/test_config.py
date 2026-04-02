"""Tests for annotateez.config."""

import copy

import pytest

from annotateez.config import (
    DEFAULT_CONFIG,
    _make_odd,
    load_config,
    merge_channels,
    save_config,
)


# --- _make_odd ---

def test_make_odd_even():
    assert _make_odd(4) == 5


def test_make_odd_odd():
    assert _make_odd(5) == 5


def test_make_odd_one():
    assert _make_odd(1) == 1


# --- save_config / load_config ---

def test_load_config_creates_default_file(tmp_path, monkeypatch):
    """load_config writes a default file when none exists."""
    monkeypatch.setattr("annotateez.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("annotateez.config.CONFIG_PATH", tmp_path / "config.yml")

    load_config()

    assert (tmp_path / "config.yml").exists()


def test_load_config_returns_default_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("annotateez.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("annotateez.config.CONFIG_PATH", tmp_path / "config.yml")

    config = load_config()

    for key in ("labels", "tile_size", "x_size", "y_size", "active_label"):
        assert key in config


def test_load_config_tile_size_always_odd(tmp_path, monkeypatch):
    """load_config corrects an even tile_size to odd."""
    monkeypatch.setattr("annotateez.config.CONFIG_DIR", tmp_path)
    cfg_path = tmp_path / "config.yml"
    monkeypatch.setattr("annotateez.config.CONFIG_PATH", cfg_path)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["tile_size"] = 10  # even
    save_config(config)

    loaded = load_config()

    assert loaded["tile_size"] % 2 == 1


def test_save_load_round_trip(tmp_path, monkeypatch):
    """Values written by save_config are restored by load_config."""
    monkeypatch.setattr("annotateez.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("annotateez.config.CONFIG_PATH", tmp_path / "config.yml")

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["x_size"] = 12
    config["y_size"] = 8
    save_config(config)

    loaded = load_config()

    assert loaded["x_size"] == 12
    assert loaded["y_size"] == 8
    assert loaded["labels"] == config["labels"]


# --- merge_channels ---

def test_merge_channels_new_channels():
    """New channels default to display_color 'none'."""
    config = {"channels": []}
    merge_channels(config, ["DAPI", "CD45"])

    assert config["channels"] == [
        {"name": "DAPI", "display_color": "none"},
        {"name": "CD45", "display_color": "none"},
    ]


def test_merge_channels_preserves_existing_colors():
    """Known channels keep their existing display_color."""
    config = {"channels": [{"name": "DAPI", "display_color": "blue"}]}
    merge_channels(config, ["DAPI", "CD45"])

    assert config["channels"][0]["display_color"] == "blue"
    assert config["channels"][1]["display_color"] == "none"


def test_merge_channels_order_follows_names():
    """Output order matches channel_names, regardless of config order."""
    config = {
        "channels": [
            {"name": "CD45", "display_color": "green"},
            {"name": "DAPI", "display_color": "blue"},
        ]
    }
    merge_channels(config, ["DAPI", "CD45"])

    names = [ch["name"] for ch in config["channels"]]
    assert names == ["DAPI", "CD45"]
    assert config["channels"][0]["display_color"] == "blue"
    assert config["channels"][1]["display_color"] == "green"


def test_merge_channels_empty_names():
    """Empty channel list clears config channels."""
    config = {"channels": [{"name": "DAPI", "display_color": "blue"}]}
    merge_channels(config, [])

    assert config["channels"] == []
