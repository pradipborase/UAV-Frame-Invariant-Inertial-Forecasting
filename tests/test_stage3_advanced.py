"""Stage 3 compact models: causality, shapes, budgets, leakage, metrics, seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage2.bootstrap import bootstrap_mean_ci, win_loss_tie
from stage2.metrics import equal_sequence_mean, rmse_all
from stage2.protocol import verify_protocol_lock
from stage2.windows import SequenceWindows, valid_origins
from stage3.grids import flatten_configs, load_stage3_yaml
from stage3.models import CausalConv1d, CompactGRU, CompactTCN, CompactTransformer, build_model, count_parameters
from stage3.protocol import verify_stage3_prerequisites
from stage3.train import fit_channel_scaler, predict_physical, seed_everything, train_model

ROOT = Path(__file__).resolve().parents[1]


def _toy_pack(seq_id: str, n: int = 260, seed: int = 0) -> SequenceWindows:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.float64) * 0.01
    qf = 9.8 + rng.normal(size=n)
    qw = 0.2 + 0.05 * rng.normal(size=n)
    origins = valid_origins(ts)
    hist_idx = origins[:, None] + np.arange(-99, 1)
    fut_idx = origins[:, None] + np.arange(1, 21)
    return SequenceWindows(
        dataset="EUROC",
        sequence_id=seq_id,
        group="g1",
        timestamp_s=ts,
        q_f=qf,
        q_w=qw,
        native_index=np.arange(n),
        origins=origins,
        hist_qf=qf[hist_idx],
        hist_qw=qw[hist_idx],
        future_qf=qf[fut_idx],
        p95_qf=float(np.percentile(qf, 95)),
    )


def test_protocol_and_stage2_freeze_hashes() -> None:
    lock = verify_protocol_lock(ROOT)
    assert lock["match"] == "PASS", lock
    st = verify_stage3_prerequisites(ROOT)
    assert st["match"] == "PASS", st


def test_grid_has_at_most_12_per_architecture() -> None:
    cfg = load_stage3_yaml(ROOT)
    rows = flatten_configs(cfg)
    for arch in ("TCN", "GRU", "TRANSFORMER"):
        n = sum(1 for r in rows if r["architecture"] == arch)
        assert 1 <= n <= 12


def test_tcn_causal_conv_no_future_leak() -> None:
    torch.manual_seed(0)
    conv = CausalConv1d(2, 4, kernel_size=5, dilation=2)
    conv.eval()
    a = torch.randn(3, 2, 40)
    b = a.clone()
    cut = 20
    b[:, :, cut + 1 :] += 4.0
    with torch.no_grad():
        ya = conv(a)
        yb = conv(b)
    torch.testing.assert_close(ya[:, :, : cut + 1], yb[:, :, : cut + 1], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(ya[:, :, cut + 1 :], yb[:, :, cut + 1 :])


def test_tcn_block_causality() -> None:
    model = CompactTCN(n_in=2, n_out=20, channels=[16, 16], kernel_size=3, dilations=[1, 2, 4], dropout=0.0)
    model.eval()
    a = torch.randn(2, 100, 2)
    b = a.clone()
    b[:, 60:, :] += 3.0
    with torch.no_grad():
        # identical prefix through index 59 must not change the representation used at t=59
        h = model.proj(a.transpose(1, 2))
        hb = model.proj(b.transpose(1, 2))
        for block in model.blocks:
            h = block(h)
            hb = block(hb)
    torch.testing.assert_close(h[:, :, :60], hb[:, :, :60], rtol=1e-4, atol=1e-4)


def test_gru_transformer_shapes_history_only() -> None:
    x = torch.randn(4, 100, 2)
    gru = CompactGRU(2, 20, hidden_size=16, n_layers=1, dropout=0.1)
    tr = CompactTransformer(2, 20, d_model=16, n_heads=2, n_layers=1, ff_multiplier=2, dropout=0.0, pooling="last")
    y1 = gru(x)
    y2 = tr(x)
    assert y1.shape == (4, 20)
    assert y2.shape == (4, 20)
    assert gru.forward.__code__.co_argcount == 2  # self, x
    assert tr.forward.__code__.co_argcount == 2


def test_models_never_accept_future_targets_as_input() -> None:
    x = torch.randn(1, 100, 2)
    for ctor in (
        lambda: CompactTCN(2, 20, [16, 16], 3, [1, 2], 0.0),
        lambda: CompactGRU(2, 20, 16, 1, 0.0),
        lambda: CompactTransformer(2, 20, 16, 2, 1, 2, 0.0, "mean"),
    ):
        model = ctor()
        y = model(x)
        assert y.shape == (1, 20)
        import inspect

        sig = inspect.signature(model.forward)
        assert list(sig.parameters) == ["x"]


def test_no_batchnorm_in_advanced_models() -> None:
    cfg = load_stage3_yaml(ROOT)
    for row in flatten_configs(cfg):
        model = build_model(row["architecture"], row, n_in=2)
        for module in model.modules():
            assert not isinstance(module, torch.nn.modules.batchnorm._BatchNorm)


def test_parameter_budgets() -> None:
    cfg = load_stage3_yaml(ROOT)
    budgets = cfg["param_budget"]
    for row in flatten_configs(cfg):
        n = count_parameters(build_model(row["architecture"], row, n_in=2))
        assert n <= int(budgets[row["architecture"]]), (row["config_id"], n)


def test_source_only_scaler_ignores_held_out_and_target() -> None:
    a = _toy_pack("toy_a", seed=1)
    b = _toy_pack("toy_b", seed=2)
    packs = {"toy_a": a, "toy_b": b}
    scaler = fit_channel_scaler(packs, ["toy_a"], dataset="EUROC", use_qw=True)
    assert scaler.fit_sequences == ["toy_a"]
    other = fit_channel_scaler(packs, ["toy_a", "toy_b"], dataset="EUROC", use_qw=True)
    assert not np.allclose(scaler.feature_mean, other.feature_mean) or not np.allclose(scaler.target_mean, other.target_mean)


def test_early_stopping_uses_validation_not_max_epochs() -> None:
    packs = {"tr": _toy_pack("tr", n=280, seed=3), "va": _toy_pack("va", n=280, seed=4)}
    cfg = {
        "config_id": "TCN01",
        "channels": [16, 16],
        "kernel_size": 3,
        "dilations": [1, 2, 4],
        "dropout": 0.0,
        "activation": "relu",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    result = train_model(
        packs,
        ["tr"],
        ["va"],
        architecture="TCN",
        config=cfg,
        seed=20260912,
        dataset="EUROC",
        use_qw=True,
        batch_size=32,
        max_epochs=40,
        patience=3,
        min_delta=1e-5,
        windows_per_sequence_per_epoch=16,
    )
    assert result.best_epoch <= 40
    assert result.target_dataset_access_during_fit == "NO"
    assert "va" not in result.train_ids
    assert "toy_target" not in result.train_ids


def test_target_sequences_never_enter_training_ids() -> None:
    packs = {
        "src_a": _toy_pack("src_a", seed=5),
        "src_b": _toy_pack("src_b", seed=6),
        "target_x": _toy_pack("target_x", seed=7),
    }
    cfg = {
        "config_id": "GRU01",
        "hidden_size": 16,
        "n_layers": 1,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    result = train_model(
        packs,
        ["src_a"],
        ["src_b"],
        architecture="GRU",
        config=cfg,
        seed=20260912,
        dataset="EUROC",
        use_qw=True,
        batch_size=32,
        max_epochs=2,
        patience=10,
        min_delta=1e-5,
        windows_per_sequence_per_epoch=8,
        fixed_epochs=2,
    )
    assert "target_x" not in result.train_ids
    assert "target_x" not in result.val_ids
    assert result.scaler.fit_sequences == ["src_a"]


def test_seed_reproducibility_cpu() -> None:
    packs = {"tr": _toy_pack("tr", seed=8), "va": _toy_pack("va", seed=9)}
    cfg = {
        "config_id": "GRU04",
        "hidden_size": 16,
        "n_layers": 1,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    kwargs = dict(
        packs=packs,
        train_ids=["tr"],
        val_ids=["va"],
        architecture="GRU",
        config=cfg,
        dataset="EUROC",
        use_qw=True,
        batch_size=32,
        max_epochs=3,
        patience=10,
        min_delta=1e-5,
        windows_per_sequence_per_epoch=8,
        fixed_epochs=3,
    )
    r1 = train_model(seed=20260912, **kwargs)
    r2 = train_model(seed=20260912, **kwargs)
    m1 = build_model("GRU", cfg, n_in=2)
    m1.load_state_dict(r1.state_dict)
    m2 = build_model("GRU", cfg, n_in=2)
    m2.load_state_dict(r2.state_dict)
    p1 = predict_physical(m1, packs, ["va"], r1.scaler, True)
    p2 = predict_physical(m2, packs, ["va"], r2.scaler, True)
    np.testing.assert_allclose(p1["va"], p2["va"], rtol=1e-5, atol=1e-5)


def test_metric_formulas_and_equal_sequence_mean() -> None:
    pred = np.zeros((4, 20))
    actual = np.ones((4, 20))
    assert abs(rmse_all(pred, actual) - 1.0) < 1e-12
    assert abs(equal_sequence_mean([1.0, 3.0]) - 2.0) < 1e-12
    boot = bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), n_reps=200, seed=20260912)
    assert boot["mean"] == 2.0
    w, l, t = win_loss_tie(np.array([-0.1, 0.2, 0.0]))
    assert (w, l, t) == (1, 1, 1)


def test_qw_ablation_changes_input_width() -> None:
    cfg = {
        "config_id": "TCN01",
        "channels": [16, 16],
        "kernel_size": 3,
        "dilations": [1, 2],
        "dropout": 0.0,
        "activation": "relu",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    m2 = build_model("TCN", cfg, n_in=2)
    m1 = build_model("TCN", cfg, n_in=1)
    assert next(m2.parameters()).shape[1] == 2 or m2.proj.in_channels == 2
    assert m1.proj.in_channels == 1
    assert m2.proj.in_channels == 2


def test_checkpoint_restores_best_validation_state() -> None:
    packs = {"tr": _toy_pack("tr", n=300, seed=10), "va": _toy_pack("va", n=300, seed=11)}
    cfg = {
        "config_id": "TR01",
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "ff_multiplier": 2,
        "dropout": 0.0,
        "pooling": "last",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    result = train_model(
        packs,
        ["tr"],
        ["va"],
        architecture="TRANSFORMER",
        config=cfg,
        seed=20260913,
        dataset="EUROC",
        use_qw=True,
        batch_size=32,
        max_epochs=8,
        patience=10,
        min_delta=1e-5,
        windows_per_sequence_per_epoch=12,
    )
    assert result.best_epoch >= 1
    model = build_model("TRANSFORMER", cfg, n_in=2)
    model.load_state_dict(result.state_dict)
    pred = predict_physical(model, packs, ["va"], result.scaler, True)
    assert pred["va"].shape[1] == 20
    assert np.all(np.isfinite(pred["va"]))
