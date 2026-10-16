import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from rag import local_backend


@pytest.fixture
def local_server(monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body))
            reply = ({'data': [{'embedding': [.1, .2, .3]}]}
                     if self.path == '/v1/embeddings'
                     else {'choices': [{'message': {'role': 'assistant', 'content': 'Grounded answer [doc]'}}]})
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('CYBERRAG_BACKEND', 'llamacpp')
    for kind in ['GEN', 'EMBED']:
        monkeypatch.setenv(f'CYBERRAG_{kind}_URL', f'http://127.0.0.1:{server.server_port}')
    yield requests
    server.shutdown()
    thread.join()
    server.server_close()


def test_local_embedding_wire_contract(local_server):
    result = local_backend.embeddings('nomic', 'T1059')
    assert result['embedding'] == [.1, .2, .3]
    assert local_server[0] == ('/v1/embeddings', {'model': 'nomic', 'input': 'T1059'})


def test_generation_preserves_prompt_limits_and_citations(local_server):
    result = local_backend.generate('local', 'Explain the source', options={'num_predict': 17, 'temperature': 0})
    assert result['response'] == 'Grounded answer [doc]'
    _, request = local_server[0]
    assert request['messages'] == [{'role': 'user', 'content': 'Explain the source'}]
    assert request['max_tokens'] == 17
    assert request['temperature'] == 0
    assert request['stream'] is False
    assert request['chat_template_kwargs'] == {'enable_thinking': False}


def test_empty_model_answer_is_an_error(monkeypatch):
    monkeypatch.setenv('CYBERRAG_BACKEND', 'llamacpp')
    monkeypatch.setattr(local_backend, '_post', lambda *args: {
        'choices': [{'message': {'content': ''}}]})
    with pytest.raises(RuntimeError, match='no answer'):
        local_backend.chat('local', [{'role': 'user', 'content': 'T1059'}])


@pytest.mark.parametrize('url', ['https://example.com', 'http://192.0.2.1',
                                'http://user:password@localhost:8088'])
def test_local_mode_rejects_external_or_credential_urls(monkeypatch, url):
    monkeypatch.setenv('CYBERRAG_BACKEND', 'llamacpp')
    monkeypatch.setenv('CYBERRAG_GEN_URL', url)
    with pytest.raises(ValueError, match='loopback'):
        local_backend.chat('local', [{'role': 'user', 'content': 'private fixture'}])


def test_existing_ollama_default_remains_available(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.delenv('CYBERRAG_BACKEND', raising=False)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(embeddings=lambda **kw: {'embedding': [1]}))
    assert local_backend.embeddings('existing', 'hello') == {'embedding': [1]}
