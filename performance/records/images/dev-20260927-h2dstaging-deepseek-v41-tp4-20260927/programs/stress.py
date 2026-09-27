"""Concurrent correctness probe for intermittent output corruption.

Usage: BASE_URL=http://host:port MODEL=name [API_KEY=...] python3 stress.py LABEL ROUNDS

Each round sends 24 short greedy questions with known answers and 8 questions
about a vault code hidden in about 6K tokens of filler, all 32 through a pool
of 16 threads, so prefill, speculative decoding and asynchronous scheduling
of different requests interleave. Thinking is disabled. A response is
degenerate when one word repeats at least 8 times in a row; other responses
that miss the expected answer count as wrong. The last output line is a JSON
summary.
"""
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import urllib.request

KEY = os.environ.get('API_KEY', '')
URL = os.environ.get('BASE_URL', 'http://127.0.0.1:8888') + '/v1/chat/completions'
MODEL = os.environ['MODEL']
LABEL = sys.argv[1] if len(sys.argv) > 1 else 'x'
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
Q = [
 ("a1", "What is 347 * 29? Reply with the number only.", r"\b10063\b"),
 ("a2", "What is 2^20? Reply with the number only.", r"\b1048576\b"),
 ("a3", "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Number only.", r"\b72\b"),
 ("a4", "If x + 2x + 3x = 48, what is x? Number only.", r"\b8\b"),
 ("a5", "How many prime numbers are there below 50? Number only.", r"\b15\b"),
 ("a7", "What is the remainder when 1000 is divided by 7? Number only.", r"\b6\b"),
 ("a8", "Compute 15% of 240. Number only.", r"\b36\b"),
 ("f1", "What is the capital of Australia? One word.", r"Canberra"),
 ("f2", "Which element has the chemical symbol 'Fe'? One word.", r"[Ii]ron"),
 ("f3", "Who wrote 'Pride and Prejudice'? Name only.", r"Austen"),
 ("f4", "What is the largest planet in our solar system? One word.", r"Jupiter"),
 ("f5", "In what year did the Berlin Wall fall? Number only.", r"1989"),
 ("f6", "中国的首都是哪里？只回答城市名。", r"北京"),
 ("f7", "水的化学式是什么？只回答化学式。", r"H\s*2\s*O|H₂O"),
 ("c1", "What does this Python print? print(sorted([3,1,2], reverse=True)[1])  Answer with the output only.", r"^\s*`*2`*\s*$"),
 ("c2", "What does this Python print? print(len({1,2,2,3,3,3}))  Answer with the output only.", r"^\s*`*3`*\s*$"),
 ("c3", "What does this Python print? print('abc'[::-1])  Answer with the output only.", r"cba"),
 ("c4", "Write a Python function is_palindrome(s) that returns True if s reads the same backwards. Code only.", r"def is_palindrome"),
 ("c5", "Write a Python function fib(n) returning the n-th Fibonacci number iteratively. Code only.", r"def fib"),
 ("l1", "All cats are mammals. Tom is a cat. Is Tom a mammal? Answer yes or no.", r"(?i)\byes\b"),
 ("l2", "If today is Wednesday, what day will it be in 10 days? One word.", r"Saturday"),
 ("l4", "How many letters 'r' are in the word 'strawberry'? Number only.", r"\b3\b"),
 ("t1", "Translate to English: '我今天很高兴见到你。' Answer with the translation only.", r"(?i)happy|glad|pleased"),
 ("t2", "Explain in two sentences why the sky is blue.", r"(?i)scatter"),
]
WORDS = ("the river keeps moving past old stone bridges while merchants count copper coins and children chase "
         "gulls along the harbor wall as clouds gather over distant hills and lanterns flicker in narrow streets").split()
def long_prompt(seed):
    rnd = random.Random(seed)
    code = str(10000 + seed * 7919 % 89999)
    parts = []
    for i in range(60):
        parts.append(' '.join(rnd.choice(WORDS) for _ in range(70)) + '.')
        if i == 30:
            parts.append(f'Important: the vault access code is {code}.')
    return (f"L{seed}", '\n'.join(parts) + '\n\nWhat is the vault access code mentioned above? Reply with the number only.',
            r'\b' + code + r'\b')
DEGEN = re.compile(r'(\b\S+\b)(?:\s+\1\b){7,}')
def ask(item):
    qid, prompt, pat = item
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        t = json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return {"id": qid, "ok": False, "degen": False, "error": repr(e)[:200], "text": ""}
    return {"id": qid, "ok": bool(re.search(pat, t.strip(), re.M)), "degen": bool(DEGEN.search(t)), "text": t[:240]}
allres = []
for rd in range(ROUNDS):
    items = Q + [long_prompt(rd * 8 + j + 1) for j in range(8)]
    random.Random(rd).shuffle(items)
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(ask, items))
    for x in res:
        x['round'] = rd
    allres += res
bad = [x for x in allres if x['degen']]
wrong = [x for x in allres if not x['ok'] and not x['degen']]
errs = [x for x in allres if x.get('error')]
print(json.dumps({"label": LABEL, "n": len(allres), "degen": len(bad), "wrong": len(wrong), "errors": len(errs),
                  "degen_items": [(x['round'], x['id'], x['text'][:120]) for x in bad],
                  "wrong_ids": sorted({x['id'] for x in wrong}), "wrong_samples": [(x['id'], x['text'][:80]) for x in wrong[:8]]}))
