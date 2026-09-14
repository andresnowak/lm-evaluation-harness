import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.task import Task
from lm_eval.models.megatron_lm import MegatronLMEval


class _LikelihoodTask(Task):
    VERSION = 0
    OUTPUT_TYPE = "loglikelihood"
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, name, count):
        self._name = name
        self._docs = [{"idx": i} for i in range(count)]
        super().__init__(
            config={
                "task": name,
                "output_type": self.OUTPUT_TYPE,
                "num_fewshot": 0,
                "repeats": 1,
            }
        )

    @property
    def task_name(self):
        return self._name

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        del data_dir, cache_dir, download_mode
        self.dataset = None

    @property
    def eval_docs(self):
        return self._docs

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._docs

    def doc_to_text(self, doc):
        return "a" * (doc["idx"] + 1)

    def doc_to_target(self, doc):
        del doc
        return "b"

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        del num_fewshot, kwargs
        return self.doc_to_text(doc)

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            "loglikelihood", doc, (ctx, "b"), 0, metadata=kwargs["metadata"]
        )

    def process_results(self, doc, results):
        del doc
        logprob, greedy = results[0]
        return {"logprob": logprob, "greedy": float(greedy)}

    def aggregation(self):
        return {"logprob": sum, "greedy": sum}

    def higher_is_better(self):
        return {"logprob": True, "greedy": True}


class _Tokenizer:
    eod = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) - ord("a") + 2 for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token - 2 + ord("a")) for token in tokens if token >= 2)


class _InferenceModel(torch.nn.Module):
    def forward(self, input_ids, position_ids, attention_mask):
        del position_ids, attention_mask
        return torch.zeros((*input_ids.shape, 32), device=input_ids.device)


@pytest.fixture
def four_rank_process_group():
    """Skip before rendezvous unless invoked under a four-rank torchrun."""
    import torch.distributed as dist

    if (
        os.environ.get("RANK") is None
        or os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("MASTER_ADDR") is None
        or os.environ.get("MASTER_PORT") is None
    ):
        pytest.skip("requires torchrun with WORLD_SIZE=4")
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")

    initialized_here = False
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=2))
        initialized_here = True
    if dist.get_world_size() != 4:
        if initialized_here:
            dist.destroy_process_group()
        pytest.skip("requires a four-rank process group")
    try:
        yield dist
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def _require_megatron():
    path = os.environ.get("MEGATRON_PATH")
    if not path:
        pytest.skip("MEGATRON_PATH must point to Megatron-LM")
    if path not in sys.path:
        sys.path.insert(0, path)


def _device(dist):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _adapter(dist, device, *, ep_size=1, batch_size=2):
    model = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(model)
    model.tokenizer = _Tokenizer()
    model._device = device
    model._global_rank = dist.get_rank()
    model._max_length = 16
    model._batch_size = batch_size
    model._tp_size = 1
    model._ep_size = ep_size
    model._parallelism_mode = "data_parallel"
    model._args = SimpleNamespace(sequence_parallel=False)
    model._set_parallelism(dist.get_world_size())
    return model


def _fake_next_token_forward(dist, ep_group):
    def forward(input_ids, attention_mask=None):
        del attention_mask
        marker = torch.ones((), device=input_ids.device)
        dist.all_reduce(marker, group=ep_group)
        logits = torch.zeros((*input_ids.shape, 32), device=input_ids.device)
        for batch in range(input_ids.shape[0]):
            for position in range(input_ids.shape[1] - 1):
                logits[batch, position, input_ids[batch, position + 1]] = 10
        return logits

    return forward


def test_ep_dp_shards_keep_collectives_aligned(four_rank_process_group):
    """Uneven and empty evaluator shards must still enter EP collectives."""
    dist = four_rank_process_group
    _require_megatron()
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        order="tp-ep-dp-pp",
    )
    try:
        device = _device(dist)
        ep_group = parallel_state.get_expert_model_parallel_group()
        for count in (dist.get_world_size() + 1, dist.get_world_size() // 2):
            model = _adapter(dist, device, ep_size=2)
            model._model_forward = _fake_next_token_forward(dist, ep_group)
            name = f"ep_shard_{count}"
            result = evaluator.evaluate(
                lm=model,
                task_dict={name: _LikelihoodTask(name, count)},
                limit=count,
                bootstrap_iters=0,
                log_samples=False,
            )
            if dist.get_rank() == 0:
                assert result is not None
                assert result["results"][name]["greedy,none"] == count
            else:
                assert result is None
    finally:
        parallel_state.destroy_model_parallel()


def test_router_metric_collectives_align_with_empty_shards(four_rank_process_group):
    """Metric collectors must run equally often on every evaluator rank."""
    dist = four_rank_process_group
    device = _device(dist)
    for count in (dist.get_world_size() + 1, dist.get_world_size() // 2):
        model = _adapter(dist, device)
        model.model = _InferenceModel()
        model._args = SimpleNamespace(
            sequence_parallel=False,
            moe_router_inference_violation_metrics=["mbs", "seq"],
        )
        calls = 0

        def collect(received, pg_collection=None):
            nonlocal calls
            del received, pg_collection
            calls += 1
            marker = torch.ones((), device=device)
            dist.all_reduce(marker)
            assert marker.item() == dist.get_world_size()
            return {}

        model._inference_metric_collectors = [collect]
        name = f"metric_shard_{count}"
        result = evaluator.evaluate(
            lm=model,
            task_dict={name: _LikelihoodTask(name, count)},
            limit=count,
            bootstrap_iters=0,
            log_samples=False,
        )
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, calls)
        assert len(set(counts)) == 1 and counts[0] > 0
        if dist.get_rank() == 0:
            assert result is not None
        else:
            assert result is None


def test_real_tp_ep_sequence_parallel_model(four_rank_process_group):
    """Exercise one real TP=2/EP=2 all-to-all model with evaluator DP."""
    dist = four_rank_process_group
    _require_megatron()
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        order="tp-ep-dp-pp",
    )
    try:
        device = _device(dist)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            moe_ffn_hidden_size=32,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_token_dispatcher_type="alltoall",  # noqa: S106
            moe_grouped_gemm=False,
            moe_aux_loss_coeff=0.0,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            params_dtype=torch.float32,
            transformer_impl="local",
        )
        megatron_model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(
                num_experts=config.num_moe_experts, moe_grouped_gemm=False
            ),
            vocab_size=32,
            max_sequence_length=16,
            parallel_output=False,
        ).to(device)
        megatron_model.eval()

        adapter = _adapter(dist, device, ep_size=2)
        adapter.model = megatron_model
        adapter._tp_size = 2
        adapter._args = SimpleNamespace(sequence_parallel=True)
        adapter._dp_rank = parallel_state.get_data_parallel_rank()
        adapter._dp_world_size = parallel_state.get_data_parallel_world_size()
        adapter._dp_group = parallel_state.get_data_parallel_group()
        adapter._set_parallelism(dist.get_world_size())
        name = "real_tp_ep"
        result = evaluator.evaluate(
            lm=adapter,
            task_dict={name: _LikelihoodTask(name, 3)},
            limit=3,
            bootstrap_iters=0,
            log_samples=False,
        )
        if dist.get_rank() == 0:
            assert result is not None
            assert torch.isfinite(torch.tensor(result["results"][name]["logprob,none"]))
        else:
            assert result is None
    finally:
        parallel_state.destroy_model_parallel()
