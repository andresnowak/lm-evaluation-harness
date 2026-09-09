from types import SimpleNamespace

import torch

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models import megatron_lm
from lm_eval.models.megatron_lm import MegatronLMEval


class _GatherCounts:
    def __init__(self, counts: list[int], remote_example=None):
        self.counts = counts
        self.remote_example = remote_example

    def gather(self, local_count: torch.Tensor) -> torch.Tensor:
        assert local_count.item() == self.counts[0]
        return torch.tensor(self.counts, device=local_count.device)

    def gather_object(self, local_example):
        return [local_example, self.remote_example]


class _RollingModel(MegatronLMEval):
    @property
    def accelerator(self):
        return self._test_accelerator


def _model(counts: list[int], remote_example=None) -> _RollingModel:
    model = _RollingModel.__new__(_RollingModel)
    LM.__init__(model)
    model._global_rank = 0
    model._rank = 0
    model._world_size = len(counts)
    model._tp_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)
    model._device = torch.device("cpu")
    model._max_length = 4
    model._test_accelerator = _GatherCounts(counts, remote_example)
    model.tokenizer = SimpleNamespace(eod=0)
    model.tok_encode = lambda string: list(range(len(string)))
    return model


def test_rolling_likelihood_flattens_and_pads_windows(monkeypatch):
    model = _model([3, 5])
    calls = []
    cached = []
    model.cache_hook.add_partial = lambda *args: cached.append(args)

    monkeypatch.setattr(
        megatron_lm,
        "get_rolling_token_windows",
        lambda token_list, **kwargs: [([0], [1])] * len(token_list),
    )
    monkeypatch.setattr(megatron_lm, "make_disjoint_window", lambda window: window)

    def score(windows, disable_tqdm=False):
        calls.append((windows, disable_tqdm))
        return [(1.0, False)] * len(windows)

    model._loglikelihood_tokens = score
    requests = [
        Instance(
            request_type="loglikelihood_rolling", doc={}, arguments=("ab",), idx=0
        ),
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=("c",), idx=1),
    ]

    assert model.loglikelihood_rolling(requests) == [2.0, 1.0]
    assert len(calls) == 1
    assert len(calls[0][0]) == 5
    assert calls[0][0][-2:] == [(None, [0], [1]), (None, [0], [1])]
    assert [entry[1] for entry in cached] == [("ab",), ("c",)]


def test_rolling_likelihood_pads_an_empty_local_shard(monkeypatch):
    remote_window = (None, [9], [10])
    model = _model([0, 2], remote_example=remote_window)
    scored = []
    model._loglikelihood_tokens = lambda windows, disable_tqdm=False: scored.extend(
        windows
    ) or [(0.0, False)] * len(windows)

    monkeypatch.setattr(
        megatron_lm,
        "get_rolling_token_windows",
        lambda token_list, **kwargs: [],
    )

    assert model.loglikelihood_rolling([]) == []
    assert scored == [remote_window, remote_window]
