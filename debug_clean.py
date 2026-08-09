raw_code = '''import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (offs_k + k_start)[None, :] * stride_ak)
        b = tl.load(b_ptr + (offs_k + k_start)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    c = acc.to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

def kernel(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C
'''

print('=== Step 0: Original ===')
print(repr(raw_code[:200]))
print()

code = raw_code.strip()
print('=== Step 1: After strip ===')
print(repr(code[:200]))
print()

for end_tag in ['</think>', '</thinking>']:
    idx = code.find(end_tag)
    if idx != -1:
        code = code[idx + len(end_tag):]
        break
print('=== Step 2: After think tags ===')
print(repr(code[:200]))
print()

if code.startswith('```'):
    lines = code.split('\n')
    start = 1
    if start < len(lines) and lines[start].strip().startswith(('python', 'triton', 'cpp', 'cuda', 'c++')):
        start += 1
    code = '\n'.join(lines[start:])
print('=== Step 3: After code block start ===')
print(repr(code[:200]))
print()

if code.rstrip().endswith('```'):
    code = code.rstrip()[:-3]
print('=== Step 4: After code block end ===')
print(repr(code[:200]))
print()

valid_start_markers = ('import triton', 'import torch', 'from triton', '@triton.jit')
found_idx = -1
for marker in valid_start_markers:
    idx = code.find(marker)
    if idx >= 0 and (found_idx == -1 or idx < found_idx):
        found_idx = idx
if found_idx >= 0:
    code = code[found_idx:]
print('=== Step 5: After finding start marker ===')
print(repr(code[:200]))
print()

lines = code.split('\n')
cleaned_lines = []
in_code = False
for line in lines:
    stripped = line.strip()
    
    if stripped.startswith(valid_start_markers):
        in_code = True
        
    if in_code:
        if stripped == '```':
            break
        if stripped.startswith('**') and not '@triton' in stripped and not 'def ' in stripped and not 'import' in stripped:
            break
        if stripped.startswith('*   ') or stripped.startswith('    *   '):
            continue
        if stripped.startswith('    *') and len(stripped) > 6 and not stripped[6].isspace():
            continue
        if '`' in stripped and ('import' in stripped or 'def ' in stripped or '@triton' in stripped):
            stripped = stripped.replace('`', '')
            line = ' ' * (len(line) - len(stripped)) + stripped
        cleaned_lines.append(line)
print('=== Step 6: After line filtering ===')
result = '\n'.join(cleaned_lines)
print(repr(result[:200]))
print()

code = result.strip()
print('=== Final result ===')
print(code)
