import pyarrow  # MUST be first — Windows DLL fix (pyarrow must load before CUDA DLLs)
import os
import re
import types as _types
import transformers
import torch
from transformer_lens import HookedTransformer

_model_cache: dict = {}

LARGE_MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "google/gemma-2-9b-it",
]

SUPPORTED_MODELS: list[str] = ["gpt2", "gpt2-medium"] + LARGE_MODELS

_LARGE_MODELS_SET: set[str] = set(LARGE_MODELS)


def is_large_model(model_name: str) -> bool:
    return model_name in _LARGE_MODELS_SET


def check_vram_available(required_gb: float = 6.0) -> bool:
    """
    Check that at least required_gb of free VRAM is available.
    Raises RuntimeError if CUDA unavailable or VRAM insufficient.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — cannot load large model.")
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {free_gb:.1f} GB free, {required_gb:.1f} GB required."
        )
    return True


def _parse_resid_post_hook(hook_name: str) -> int:
    """Extract layer index from 'blocks.{layer}.hook_resid_post'."""
    m = re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", hook_name)
    if m is None:
        raise ValueError(f"Unsupported hook name for LargeModelWrapper: {hook_name!r}")
    return int(m.group(1))


def _make_pt_hook(hook_fn):
    """
    Wrap a TL-style hook_fn(act, hook=None) into a PyTorch forward hook.

    transformers 4.x: LlamaDecoderLayer returns a tuple (hidden_states, ...).
    transformers 5.x: LlamaDecoderLayer returns a plain Tensor (batch, seq, d_model).
    Both cases are handled: hidden_states is extracted and passed to hook_fn as-is
    (shape stays (batch, seq, d_model) matching TL convention).
    Returning None from hook_fn = capture only (probing);
    returning a Tensor = replace hidden states (steering).
    """
    def pt_hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        hidden = output if is_tensor else output[0]  # (batch, seq, d_model)
        result = hook_fn(hidden, hook=None)
        if result is not None and isinstance(result, torch.Tensor):
            return result if is_tensor else (result,) + output[1:]
        return output
    return pt_hook


class LargeModelWrapper:
    """
    Thin wrapper around a 4-bit quantized HuggingFace model exposing the
    TransformerLens interface so that probing and steering code is model-agnostic.

    Supported interface:
        cfg.n_layers, cfg.d_model, cfg.d_vocab, cfg.model_name
        to_tokens(text) -> Tensor(1, seq)
        to_string(tokens) -> str
        __call__(tokens) -> logits Tensor(batch, seq, vocab)
        run_with_hooks(tokens, fwd_hooks) -> logits Tensor(batch, seq, vocab)
        add_hook(hook_name, hook_fn)   — persistent hook; used by apply_steering
        reset_hooks()                   — remove all persistent hooks
        parameters()                    — delegate to HF model

    Hook name format: 'blocks.{layer}.hook_resid_post' only.
    Hook fn signature: fn(act: Tensor(batch, seq, d_model), hook=None) -> Tensor | None.
    TODO: hook_resid_pre, hook_attn_out, hook_mlp_out not supported — use HookedTransformer.
    """

    def __init__(self, hf_model, tokenizer, device: str) -> None:
        self._model = hf_model
        self._tokenizer = tokenizer
        self._device = device
        self._persistent_hooks: list = []

        config = hf_model.config
        self.cfg = _types.SimpleNamespace(
            n_layers=config.num_hidden_layers,
            d_model=config.hidden_size,
            d_vocab=config.vocab_size,
            model_name=getattr(config, "_name_or_path", ""),
        )

    def to_tokens(self, text: str) -> torch.Tensor:
        enc = self._tokenizer(text, return_tensors="pt")
        return enc["input_ids"].to(self._device)

    def to_string(self, tokens: torch.Tensor) -> str:
        if tokens.dim() == 2:
            tokens = tokens[0]
        return self._tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self._model(input_ids=tokens)
        return out.logits

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: list) -> torch.Tensor:
        handles = []
        for hook_name, hook_fn in fwd_hooks:
            layer_idx = _parse_resid_post_hook(hook_name)
            handle = self._model.model.layers[layer_idx].register_forward_hook(
                _make_pt_hook(hook_fn)
            )
            handles.append(handle)
        try:
            with torch.no_grad():
                out = self._model(input_ids=tokens)
        finally:
            for h in handles:
                h.remove()
        return out.logits

    def add_hook(self, hook_name: str, hook_fn) -> None:
        layer_idx = _parse_resid_post_hook(hook_name)
        handle = self._model.model.layers[layer_idx].register_forward_hook(
            _make_pt_hook(hook_fn)
        )
        self._persistent_hooks.append(handle)

    def reset_hooks(self) -> None:
        for h in self._persistent_hooks:
            h.remove()
        self._persistent_hooks.clear()

    @property
    def tokenizer(self):
        return self._tokenizer

    def parameters(self):
        return self._model.parameters()


def get_tokenizer(model_name: str = "gpt2") -> transformers.PreTrainedTokenizerBase:
    """Return HuggingFace tokenizer; uses HF_TOKEN env var for gated models."""
    hf_token = os.getenv("HF_TOKEN")
    return transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)


def get_model(model_name: str = "gpt2", device: str = "cuda"):
    """
    Load a model with process-level caching.

    Small models (GPT-2 family): HookedTransformer.from_pretrained.
    Large models (see LARGE_MODELS): AutoModelForCausalLM with 4-bit BitsAndBytes
    quantization (~5 GB VRAM), wrapped in LargeModelWrapper.

    Args:
        model_name: Model identifier (see SUPPORTED_MODELS).
        device: 'cuda' or 'cpu'.

    Returns:
        HookedTransformer (small models) or LargeModelWrapper (large models).
    """
    key = (model_name, device)
    if key in _model_cache:
        return _model_cache[key]

    if is_large_model(model_name):
        check_vram_available(required_gb=6.0)
        # Lazy imports — must not be at module level; fast tests run without bitsandbytes.
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: PLC0415

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer = get_tokenizer(model_name)
        model = LargeModelWrapper(hf_model, tokenizer, device)
    else:
        model = HookedTransformer.from_pretrained(model_name)
        model = model.to(device)

    _model_cache[key] = model
    return model
