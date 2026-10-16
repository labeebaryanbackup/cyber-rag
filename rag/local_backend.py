"""Local runtime adapter. Select llama.cpp with CYBERRAG_BACKEND=llamacpp.

The default remains Ollama for existing installations. The llama.cpp endpoints
must be loopback HTTP servers; this path does not send the corpus to a cloud API.
"""
import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _local():
    return os.getenv('CYBERRAG_BACKEND', 'ollama') == 'llamacpp'


def _post(kind, path, payload):
    port = 8089 if kind == 'EMBED' else 8088
    base = os.getenv(f'CYBERRAG_{kind}_URL', f'http://127.0.0.1:{port}').rstrip('/')
    parsed = urlparse(base)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('The llama.cpp adapter requires a loopback HTTP endpoint.')
    request = Request(base + path, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=180) as response:
            return json.load(response)
    except OSError as error:
        raise RuntimeError(f'Local {kind.lower()} server unavailable at {base}. '
                           'Start the portfolio CyberRAG launcher first.') from error


def embeddings(model, prompt, **kwargs):
    if not _local():
        import ollama
        return ollama.embeddings(model=model, prompt=prompt, **kwargs)
    response = _post('EMBED', '/v1/embeddings', {'model': model, 'input': prompt})
    return {'embedding': response['data'][0]['embedding']}


def chat(model, messages, options=None, **kwargs):
    if not _local():
        import ollama
        return ollama.chat(model=model, messages=messages, options=options, **kwargs)
    if kwargs.get('stream'):
        raise ValueError('Streaming is not used by this adapter.')
    opts = options or {}
    response = _post('GEN', '/v1/chat/completions', {
        'model': model, 'messages': messages, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': False},
        'temperature': opts.get('temperature', .1),
        'max_tokens': opts.get('num_predict', 512),
        **({'stop': opts['stop']} if 'stop' in opts else {}),
    })
    content = response['choices'][0]['message'].get('content')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('The local model returned no answer. Check its chat template and token budget.')
    return {'message': {'role': 'assistant', 'content': content}}


def generate(model, prompt, options=None, **kwargs):
    if not _local():
        import ollama
        return ollama.generate(model=model, prompt=prompt, options=options, **kwargs)
    reply = chat(model, [{'role': 'user', 'content': prompt}], options=options, **kwargs)
    return {'response': reply['message']['content']}
