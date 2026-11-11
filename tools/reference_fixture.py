#!/usr/bin/env python3
"""Independent NumPy Qwen3 MoE forward and synthetic single-file GGUF writer.

This is test tooling only. Production inference contains no Python dependency.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np

H, D, NH, NKV, E, TOP, FF, VOCAB, LAYERS = 8, 4, 4, 2, 4, 2, 12, 16, 2
EPS, THETA = 1e-6, 10000.0
TOKENS = ['<|im_start|>', '<|im_end|>', '[UNK]', 'system', 'user', 'assistant', 'You', 'are', 'a', 'helpful', 'coding', 'assistant.', 'teste', 'hello', '<|endoftext|>', 'Rust']

def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def create(path):
    rng = np.random.default_rng(20261110)
    weights = {}
    def add(name, shape, norm=False):
        a = (rng.normal(1.0, 0.05, shape) if norm else rng.normal(0, 0.15, shape)).astype('<f4')
        weights[name] = a
    add('token_embd.weight', (VOCAB, H))
    add('output.weight', (VOCAB, H))
    add('output_norm.weight', (H,), True)
    for i in range(LAYERS):
        p = f'blk.{i}.'
        for name in ['attn_norm', 'ffn_norm']:
            add(p+name+'.weight', (H,), True)
        for name in ['attn_q_norm', 'attn_k_norm']:
            add(p+name+'.weight', (D,), True)
        add(p+'attn_q.weight', (NH*D, H))
        for name in ['attn_k', 'attn_v']:
            add(p+name+'.weight', (NKV*D, H))
        add(p+'attn_output.weight', (H, NH*D))
        add(p+'ffn_gate_inp.weight', (E, H))
        for name in ['ffn_gate_exps', 'ffn_up_exps']:
            add(p+name+'.weight', (E, FF, H))
        add(p+'ffn_down_exps.weight', (E, H, FF))
    meta = {'general.architecture': 'qwen3moe', 'qwen3moe.embedding_length':H,
        'qwen3moe.block_count':LAYERS, 'qwen3moe.attention.head_count':NH,
        'qwen3moe.attention.head_count_kv':NKV, 'qwen3moe.attention.key_length':D,
        'qwen3moe.attention.value_length':D, 'qwen3moe.expert_count':E,
        'qwen3moe.expert_used_count':TOP, 'qwen3moe.expert_feed_forward_length':FF,
        'qwen3moe.context_length':32, 'qwen3moe.rope.freq_base':THETA,
        'qwen3moe.attention.layer_norm_rms_epsilon':EPS,
        'tokenizer.ggml.tokens': TOKENS, 'tokenizer.ggml.eos_token_id': 14}
    header = b'GGUF'+struct.pack('<IQQ', 3, len(weights), len(meta))
    for key, value in meta.items():
        header += string(key)
        if isinstance(value,list):header += struct.pack('<IIQ',9,8,len(value))+b''.join(string(v) for v in value)
        elif isinstance(value, str):header += struct.pack('<I',8)+string(value)
        elif isinstance(value,float):header += struct.pack('<If',6,value)
        else:header += struct.pack('<II',4,value)
    payload = b''
    for key, a in weights.items():
        payload += b'\0'*((-len(payload))%32)
        header += string(key)+struct.pack('<I',a.ndim)+struct.pack('<'+'Q'*a.ndim,*a.shape[::-1])+struct.pack('<IQ',0,len(payload))
        payload += a.tobytes()
    header += b'\0'*((-len(header))%32)
    path.write_bytes(header+payload)
    return weights

def rms(x, weight):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+EPS) * weight

def rotate(x, pos):
    angle = pos * THETA**(-np.arange(D//2, dtype=np.float32)*2/D)
    a, b = x[:,:D//2].copy(), x[:,D//2:].copy()
    return np.concatenate([a*np.cos(angle)-b*np.sin(angle), a*np.sin(angle)+b*np.cos(angle)],axis=-1)

def softmax(x):
    y = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return y/np.sum(y,axis=-1,keepdims=True)

def reference(w, tokens):
    cache = [[] for _ in range(LAYERS)]
    results = []
    for pos, token in enumerate(tokens):
        x = w['token_embd.weight'][token].copy()
        for i in range(LAYERS):
            def a(name):return w[f'blk.{i}.{name}.weight']
            z = rms(x,a('attn_norm'))
            q = rotate(rms((a('attn_q')@z).reshape(NH,D),a('attn_q_norm')),pos)
            k = rotate(rms((a('attn_k')@z).reshape(NKV,D),a('attn_k_norm')),pos)
            v = (a('attn_v')@z).reshape(NKV,D)
            cache[i].append((k,v))
            keys = np.stack([pair[0] for pair in cache[i]])
            vals = np.stack([pair[1] for pair in cache[i]])
            att = []
            for head in range(NH):
                kvhead = head//(NH//NKV)
                scores = softmax(keys[:,kvhead,:]@q[head]/np.sqrt(D))
                att.append(scores@vals[:,kvhead,:])
            x += a('attn_output')@np.concatenate(att)
            z = rms(x,a('ffn_norm'))
            router = softmax(a('ffn_gate_inp')@z)
            chosen = np.argsort(-router,kind='stable')[:TOP]
            rates = router[chosen]/np.sum(router[chosen])
            moe = np.zeros(H,dtype=np.float32)
            for e, rate in zip(chosen,rates):
                gate, up = a('ffn_gate_exps')[e]@z, a('ffn_up_exps')[e]@z
                moe += rate*(a('ffn_down_exps')[e]@(gate/(1+np.exp(-gate))*up))
            x += moe
        results.append((w['output.weight']@rms(x,w['output_norm.weight'])).tolist())
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('test-output'))
    ap.add_argument('--binary',type=Path)
    ap.add_argument('--device',choices=['cpu','vulkan'],default='cpu')
    ap.add_argument('--allow-software',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'tiny-qwen3moe.gguf';w=create(path)
    tokens=[1,5,2,9];expected=reference(w,tokens)
    (args.out/'expected-logits.json').write_text(json.dumps(expected,indent=2))
    if args.binary:
        errors=[]
        for n in range(1,len(tokens)+1):
            cmd=[str(args.binary.resolve()),'logits','--model',str(path),'--tokens',','.join(map(str,tokens[:n])),'--ctx','16','--device',args.device]
            if args.allow_software:cmd+=['--allow-software']
            result=subprocess.run(cmd,check=True,capture_output=True,text=True)
            actual=np.array(json.loads(result.stdout)['logits'])
            error=float(np.max(np.abs(actual-np.array(expected[n-1]))));errors.append(error)
            np.testing.assert_allclose(actual,expected[n-1],rtol=2e-4,atol=2e-5)
        report={'device':args.device,'tokens':tokens,'max_absolute_error_by_prefix':errors,'passed':True,'reference':'independent NumPy synthetic 2-layer Qwen3 MoE, not pretrained 30B weights'}
        (args.out/f'parity-{args.device}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
