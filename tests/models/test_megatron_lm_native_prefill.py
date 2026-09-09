import sys
from types import ModuleType, SimpleNamespace

import pytest

from lm_eval.api.model import LM
from lm_eval.models.megatron_lm import MegatronLMEval


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Engine:
    def __init__(self, result, max_sequence_length=8):
        self.result = result
        self.context = SimpleNamespace(max_sequence_length=max_sequence_length)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(merge=lambda: self.result) for _ in kwargs["prompts"]]


def _model(monkeypatch, result):
    module_name = "megatron.core.inference.sampling_params"
    module = ModuleType(module_name)
    module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, module_name, module)

    model = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(model)
    model._native_generation_engine = _Engine(result)
    model.tokenizer = SimpleNamespace(eod=0)
    return model


def test_native_likelihood_generates_one_token_for_engine_completion(monkeypatch):
    result = SimpleNamespace(
        prompt_log_probs=[-0.1, -0.2, -0.3],
        prompt_top_n_logprobs=[{0: -0.1}, {0: -0.2}, {0: -0.3}],
    )
    model = _model(monkeypatch, result)

    assert model._native_loglikelihood([[0, 1, 2, 3]], [1], [3]) == [
        pytest.approx((-0.6, True))
    ]
    params = model._native_generation_engine.calls[0]["sampling_params"]
    assert params.num_tokens_to_generate == 1


def test_native_likelihood_uses_full_model_context_when_engine_is_longer(monkeypatch):
    result = SimpleNamespace(
        prompt_log_probs=[-0.1, -0.2, -0.3],
        prompt_top_n_logprobs=[{0: -0.1}, {0: -0.2}, {0: -0.3}],
    )
    model = _model(monkeypatch, result)
    model._use_inference_engine_for_likelihood = True
    model._max_length = 4
    model._batch_size = 1
    model._global_rank = 0
    model._tp_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)

    assert model._loglikelihood_tokens(
        [(None, [0], [1, 2, 3])], disable_tqdm=True
    ) == [pytest.approx((-0.6, True))]
    assert model._native_generation_engine.calls[0]["prompts"] == [[0, 1, 2, 3]]
