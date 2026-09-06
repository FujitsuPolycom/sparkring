"""Measure temperature-controlled client TTFT with unique-prefix prompts and receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import time
import urllib.request
from urllib.parse import urlsplit
import uuid


def tokens(value):
    text = value.lower().strip()
    scale = 1024 if text.endswith('k') else 1048576 if text.endswith('m') else 1
    result = int(float(text[:-1] if scale != 1 else text) * scale)
    if not 0 < result <= 16 * 1048576:
        raise argparse.ArgumentTypeError('Context must be positive and at most 16M')
    return result


def request(url, payload, key, timeout):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(urllib.request.Request(url, data=json.dumps(payload).encode(),
                                             headers=headers), timeout=timeout)


def events(response):
    for raw in response:
        line = raw.decode('utf-8').strip()
        if not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if value == '[DONE]':
            return
        event = json.loads(value)
        if event.get('error'):
            raise ValueError('Endpoint reported a streaming error')
        yield event
    raise ValueError('Stream ended without its completion terminator')


def has_token(event):
    return any(any(choice.get('delta', {}).get(name) for name in
                   ('content', 'reasoning_content', 'reasoning')) for choice in event.get('choices', []))


def build_text(nonce, chars):
    prefix = f'Unique benchmark request {nonce}. Read the notes and reply OK.\n'
    notes = []
    length = len(prefix)
    index = 0
    while length < chars:
        line = f'Archive row {index}: section {index % 97} records library inventory and routine maintenance checks.\n'
        notes.append(line)
        length += len(line)
        index += 1
    return (prefix + ''.join(notes))[:chars]


def measure(endpoint, model, target, limit, temperature, key, timeout):
    nonce = uuid.uuid4().hex
    chars = target * 5
    messages = None
    count = 0
    for _ in range(8):
        text = build_text(nonce, chars)
        messages = [{'role': 'user', 'content': text}]
        with request(endpoint + '/tokenize', {'model': model, 'messages': messages}, key, timeout) as response:
            count = int(json.load(response)['count'])
        if count <= 0:
            raise ValueError('Tokenizer returned an invalid count')
        if abs(count - target) <= max(4, target * 0.005):
            break
        chars = max(256, round(chars * target / count))
    else:
        raise ValueError('Token calibration did not converge within 0.5% of target')
    if count + 1 > limit:
        raise ValueError('Tokenized request exceeds the configured context limit')
    payload = {'model': model, 'messages': messages, 'temperature': temperature, 'top_p': 1,
               'max_tokens': 1, 'stream': True, 'stream_options': {'include_usage': True}}
    started = time.perf_counter()
    ttft, usage = None, None
    with request(endpoint + '/v1/chat/completions', payload, key, timeout) as response:
        for event in events(response):
            if ttft is None and has_token(event):
                ttft = time.perf_counter() - started
            if event.get('usage'):
                usage = event['usage']
    if ttft is None or ttft <= 0 or not usage or not usage.get('prompt_tokens'):
        raise ValueError('Missing first-token event or authoritative usage')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'target_tokens': target, 'tokenized_prompt_tokens': count, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'nonce': nonce, 'usage': usage, 'ttft_seconds': ttft,
            'prompt_tokens_per_second': usage['prompt_tokens'] / ttft,
            'cached_tokens_reported': cached,
            'cold_prefix_confirmed': cached == 0,
            'valid': cached in (None, 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--contexts', default='8k,16k,32k,64k,128k')
    parser.add_argument('--context-limit', type=tokens, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or not 0 <= args.temperature <= 2 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error('Require positive repeats/timeout and temperature in [0,2]')
    contexts = [tokens(x) for x in args.contexts.split(',')]
    endpoint = args.endpoint.rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    url = urlsplit(endpoint)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment or url.path:
        parser.error('Endpoint must be HTTP(S), without credentials, query, fragment or extra path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.output.open('x', encoding='utf-8') as stream:
        header = {'schema': 'sparkring-prefill-probe/v1', 'model': args.model,
                  'temperature': args.temperature, 'repeats': args.repeats, 'contexts': contexts,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'measurement': 'client request submission to first content or reasoning delta; max_tokens=1',
                  'cache_policy': 'unique prefix for every request; report server cached-token evidence'}
        stream.write(json.dumps({'metadata': header}) + '\n')
        for repeat in range(1, args.repeats + 1):
            for target in contexts:
                try:
                    cell = measure(endpoint, args.model, target, args.context_limit, args.temperature,
                                   os.environ.get(args.api_key_env, ''), args.timeout)
                except Exception as error:
                    cell = {'target_tokens': target, 'valid': False, 'error': type(error).__name__}
                cell['repeat'] = repeat
                stream.write(json.dumps(cell) + '\n')
                stream.flush()
                print(json.dumps(cell), flush=True)
                failed |= not cell['valid']
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
