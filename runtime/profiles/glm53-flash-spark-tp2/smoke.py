"""Small functional checks for the public API; not a benchmark suite."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import threading
import time
import urllib.request
import zlib

MODEL = 'GLM-5.3-Flash-NVFP4-Spark'


def text_payload(index):
    marker = f'violet-{index}-6291'
    return marker, {'model': MODEL, 'messages': [{'role': 'user', 'content':
        f'Return a JSON object with one field named code whose exact string value is {marker}. No explanation.'}],
        'temperature': 0, 'max_tokens': 256}


def correct_code(text, marker):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        return json.loads(text) == {'code': marker}
    except (ValueError, TypeError):
        return False


def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def png_blue():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = (b'\x00' + b'\x00\x00\xff' * 224) * 224
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 224, 224, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--phase', choices=('text', 'long-prompt', 'image', 'video'), required=True)
    parser.add_argument('--video-file', help='MP4 depicting a solid blue field; checks video decoding and color recognition')
    args = parser.parse_args()
    if args.phase == 'video':
        parser.error('The shared-image TP2 profile disables video input')
    base = args.base.rstrip('/')
    if args.phase == 'text':
        barrier = threading.Barrier(8)
        def one(index):
            marker, payload = text_payload(index)
            barrier.wait()
            body = post(base, '/v1/chat/completions', payload)
            output = body['choices'][0]['message'].get('content') or ''
            return {'marker': marker, 'pass': correct_code(output, marker), 'output': output, 'usage': body['usage']}
        with ThreadPoolExecutor(max_workers=8) as executor:
            records = list(executor.map(one, range(8)))
        print(json.dumps({'phase': args.phase, 'results': records}), flush=True)
        assert all(r['pass'] for r in records), 'Text marker failure'
    elif args.phase == 'long-prompt':
        prompt = 'Long-prompt recall check. The verification code is amber-73091.\n' + 'The archive contains records of buildings, gardens and bridges.\n' * 700
        prompt += '\nWhat is the verification code at the beginning?\nAnswer:'
        start = time.monotonic()
        body = post(base, '/v1/completions', {'model': MODEL, 'prompt': prompt, 'temperature': 0, 'max_tokens': 32})
        text = body['choices'][0]['text']
        print(json.dumps({'phase': args.phase, 'seconds': time.monotonic()-start, 'answer_correct': 'amber-73091' in text, 'cache_restore_verified': False, 'output': text, 'usage': body['usage']}), flush=True)
        assert 'amber-73091' in text
    else:
        if args.phase == 'image':
            content = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png_blue()).decode()}}
        else:
            from pathlib import Path
            if not args.video_file:
                parser.error('--video-file is required for video')
            content = {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + base64.b64encode(Path(args.video_file).read_bytes()).decode()}}
        body = post(base, '/v1/chat/completions', {'model': MODEL, 'messages': [{'role': 'user', 'content': [content, {'type': 'text', 'text': 'What is the dominant color? Answer with just the color.'}]}], 'temperature': 0, 'max_tokens': 256})
        message = body['choices'][0]['message']
        text = message.get('content') or ''
        print(json.dumps({'phase': args.phase, 'pass': 'blue' in text.lower(), 'message': message, 'usage': body['usage']}), flush=True)
        assert 'blue' in text.lower()


if __name__ == '__main__':
    main()
