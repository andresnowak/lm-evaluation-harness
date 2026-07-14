from types import SimpleNamespace

from lm_eval.models.megatron_lm import _override_attention_mask_type


def _layer(params):
    attention = SimpleNamespace(params=params)
    return SimpleNamespace(submodules=SimpleNamespace(self_attention=attention))


def test_attention_mask_override_skips_kda_specs_without_mask_parameter():
    full_attention = _layer({"attn_mask_type": "causal"})
    kda = _layer({})
    block = SimpleNamespace(layer_specs=[kda, full_attention])

    assert _override_attention_mask_type(block, "arbitrary") == 1
    assert kda.submodules.self_attention.params == {}
    assert full_attention.submodules.self_attention.params == {
        "attn_mask_type": "arbitrary"
    }
