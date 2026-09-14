from types import SimpleNamespace

import torch

from lm_eval.api.model import LM
from lm_eval.models.megatron_lm import MegatronLMEval


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pg_collection = object()

    def forward(self, input_ids, position_ids, attention_mask):
        del position_ids, attention_mask
        return torch.zeros((*input_ids.shape, 8))


def test_model_forward_collects_backend_metrics():
    lm = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(lm)
    lm.model = _FakeModel()
    lm._global_rank = 0
    lm._ep_size = 1
    lm._args = SimpleNamespace()
    calls = []

    def collect(model, pg_collection=None):
        calls.append((model, pg_collection))
        return {"router_max": 0.5}

    lm._inference_metric_collectors = [collect]

    output = lm._model_forward(torch.tensor([[1, 2]]))

    assert output.shape == (1, 2, 8)
    assert calls == [(lm.model, lm.model.pg_collection)]
    assert lm.get_model_metrics() == [{"router_max": 0.5}]
    assert lm.requires_uniform_request_groups
