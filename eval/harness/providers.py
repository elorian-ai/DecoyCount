#!/usr/bin/env python
"""Provider-agnostic model access for the counting benchmark.

One code path: every provider below exposes an OpenAI-compatible Chat Completions endpoint, so a
model is named `provider:model_id` and nothing else in the harness changes when you switch vendor.

    --models qwen3.8-flash=dashscope:qwen3.8-flash \
             gemini-3.8-flash=google:gemini-3.8-flash \
             gpt-5.6-sol=openai:gpt-5.6-sol

Credentials come from that provider's own standard environment variable. Nothing is read from a
config file, nothing is defaulted to a host, and no key is ever written to an output artifact.

    export OPENAI_API_KEY=...        # openai
    export GEMINI_API_KEY=...        # google
    export ANTHROPIC_API_KEY=...     # anthropic
    export DASHSCOPE_API_KEY=...     # dashscope (Qwen)

If you front several vendors with your own gateway (LiteLLM, vLLM, Bedrock proxy, ...), use the
`gateway` provider and point it at your endpoint:

    export GATEWAY_BASE_URL=https://your-gateway/v1
    export GATEWAY_API_KEY=...
    --models mymodel=gateway:openai/gpt-5.6-sol
"""
import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None            # None -> read from <NAME>_BASE_URL
    key_env: str
    base_url_env: str | None = None
    supports_reasoning_effort: bool = True
    # Some OpenAI-compatible surfaces still expect the older parameter name.
    max_tokens_param: str = 'max_completion_tokens'
    notes: str = ''

PROVIDERS: dict[str, Provider] = {
    'openai': Provider(
        'openai', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'google': Provider(
        'google', 'https://generativelanguage.googleapis.com/v1beta/openai/', 'GEMINI_API_KEY',
        notes='Gemini via its OpenAI-compatible endpoint.'),
    'anthropic': Provider(
        'anthropic', 'https://api.anthropic.com/v1/', 'ANTHROPIC_API_KEY',
        supports_reasoning_effort=False, max_tokens_param='max_tokens',
        notes='Claude via its OpenAI-compatible endpoint; no reasoning_effort.'),
    'dashscope': Provider(
        'dashscope', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'DASHSCOPE_API_KEY',
        supports_reasoning_effort=False,
        notes='Qwen. Rejects reasoning_effort, and rate-limits hard: expect ~80 calls/min '
              'regardless of --workers.'),
    'gateway': Provider(
        'gateway', None, 'GATEWAY_API_KEY', base_url_env='GATEWAY_BASE_URL',
        notes='Any OpenAI-compatible endpoint you run yourself. Model id is passed through '
              'verbatim, so it must be whatever your gateway expects.'),
}

class ProviderError(RuntimeError):
    pass

def parse_model_spec(spec: str) -> tuple[str, str, str]:
    """'alias=provider:model_id' -> (alias, provider, model_id).

    'alias=model_id' with no provider is rejected on purpose: an unqualified id silently picking a
    vendor is how a run ends up unreproducible.
    """
    if '=' not in spec:
        raise ProviderError(f"model spec {spec!r} must be alias=provider:model_id")
    alias, rest = spec.split('=', 1)
    if ':' not in rest:
        raise ProviderError(
            f"model spec {spec!r} is missing a provider. Use alias=provider:model_id, "
            f"e.g. {alias}=dashscope:qwen3.8-flash. Known providers: {', '.join(PROVIDERS)}")
    provider, model_id = rest.split(':', 1)
    if provider not in PROVIDERS:
        raise ProviderError(f"unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}")
    return alias, provider, model_id

def make_client(provider: str):
    """-> (OpenAI client, Provider). Raises with an actionable message if the key is absent."""
    from openai import OpenAI
    p = PROVIDERS[provider]
    key = os.environ.get(p.key_env)
    if not key:
        raise ProviderError(
            f"{p.key_env} is not set, which is required for provider '{provider}'. "
            f"Export it in your shell; never hardcode it or commit it.")
    base = p.base_url
    if base is None:
        base = os.environ.get(p.base_url_env or '')
        if not base:
            raise ProviderError(
                f"provider '{provider}' needs {p.base_url_env} to be set to your endpoint URL.")
    return OpenAI(base_url=base, api_key=key), p

def build_request(p: Provider, model_id: str, messages, effort: str | None, max_tokens: int,
                  timeout: int) -> tuple[dict, list[str], str | None]:
    """-> (kwargs, statically_dropped, effective_effort).

    The static drop must be reported, not silent. A dashscope run launched with --effort high would
    otherwise write rows recording effort='high' while never sending the parameter, and the effort
    curve buckets on that column -- so the run lands in the `high` cell having sent no effort at all.
    """
    kw = {'model': model_id, 'messages': messages, 'timeout': timeout,
          p.max_tokens_param: max_tokens}
    dropped: list[str] = []
    effective = effort
    if effort and effort != 'none':
        if p.supports_reasoning_effort:
            kw['reasoning_effort'] = effort
        else:
            dropped.append('reasoning_effort')
            effective = 'none'          # what was actually sent, so the row is not mislabelled
    return kw, dropped, effective

def describe() -> str:
    lines = [f"{'provider':11s} {'key env var':22s} endpoint / notes"]
    for name, p in PROVIDERS.items():
        base = p.base_url or f'<${p.base_url_env}>'
        lines.append(f"{name:11s} {p.key_env:22s} {base}")
        if p.notes:
            lines.append(f"{'':34s} {p.notes}")
    return '\n'.join(lines)

if __name__ == '__main__':
    print(describe())
