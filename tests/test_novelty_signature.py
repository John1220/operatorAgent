"""候选结构签名 (doc 47) —— 先标定「能不能看见已知的那一份重复」, 再谈用它省 GPU

这个模块存在的理由是一个具体的浪费: Conductor 每轮重新生成代码就送进三级验证, 而 L3 是
一次真机基准。如果第 3 轮只是第 1 轮改了变量名, 那一次 GPU 时间什么新信息都没换来。

但「签名能不能提出来」这种问题不能靠读自己的代码回答 —— 所以每个用例都往里面灌一份
**已知**的重复或已知的差异, 检查键的方向:
  · 改名 → 必须相等 (这就是要抓的那一类)
  · 换 BLOCK / 换算子 / 多一个循环 → 必须不等 (假阳 = 丢掉一个真的更好的核, 代价远比
    多烧一次 L3 高)
最后一组是**方向检查**: doc 46 刚栽过一次「判据方向写反, 于是量具明明看见了却报看不见」。
"""
import unittest

from operator_agent.novelty import candidate_signature, signature_of

BASE = '''import triton
import triton.language as tl

@triton.jit
def kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * 64 + tl.arange(0, 64)
    acc = tl.zeros((64,), dtype=tl.float32)
    for k in range(0, K, 32):
        a = tl.load(A + off, mask=off < M, other=0.0)
        b = tl.load(B + k, mask=k < N, other=0.0)
        acc += tl.sum(a * b, axis=0)
    tl.store(C + off, acc, mask=off < M)

def run(A, B, C):
    return kernel[(1,)](A, B, C, 1024, 1024, 1024, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        num_warps=4, num_stages=2)
'''

# 与 BASE **逐字对照**写的一份「只改名」副本: 结构、常量、调用、顺序全部相同,
# 只有标识符换了。刻意写成字面量而不是用 str.replace 生成 —— 第一版用子串替换造这份
# 样本, 结果 `mask` 被换成 `makk`、`axis` 被换成 `aaxis`, 语法直接错:  fixture 自己坏了
# 的时候, 「签名提不出来」会被误读成「模块不工作」。
RENAMED = '''import triton
import triton.language as tl

@triton.jit
def the_kernel(X, Y, Z, M, N, K, TILE_M: tl.constexpr, TILE_N: tl.constexpr,
               TILE_K: tl.constexpr):
    tile_id = tl.program_id(0)
    elems = tile_id * 64 + tl.arange(0, 64)
    total = tl.zeros((64,), dtype=tl.float32)
    for kk in range(0, K, 32):
        aa = tl.load(X + elems, mask=elems < M, other=0.0)
        bb = tl.load(Y + kk, mask=kk < N, other=0.0)
        total += tl.sum(aa * bb, axis=0)
    tl.store(Z + elems, total, mask=elems < M)

def execute(X, Y, Z):
    return the_kernel[(1,)](X, Y, Z, 1024, 1024, 1024, TILE_M=64, TILE_N=64, TILE_K=32,
                            num_warps=4, num_stages=2)
'''

BLOCK_CHANGED = BASE.replace("BLOCK_M=64", "BLOCK_M=128")
OP_CHANGED = BASE.replace("a * b", "a + b")
LOOP_ADDED = BASE.replace("for k in range(0, K, 32):",
                          "for k in range(0, K, 32):\n        pass")


class TestSignatureCalibration(unittest.TestCase):
    """方向检查: 已知的那一份重复必须被看见, 已知的差异必须被看见是不同的"""

    def test_renamed_only_code_is_the_same_candidate(self):
        a = candidate_signature(BASE)["structural"]
        b = candidate_signature(RENAMED)["structural"]
        self.assertEqual(a, b, "只改名就换了键 —— 要抓的那一类重复会全部漏掉")

    def test_block_size_change_is_a_different_candidate(self):
        self.assertNotEqual(candidate_signature(BASE)["structural"],
                            candidate_signature(BLOCK_CHANGED)["structural"],
                            "BLOCK 64→128 是配方最常改的一刀, 判成重复就是把改进丢了")

    def test_operator_change_is_a_different_candidate(self):
        self.assertNotEqual(candidate_signature(BASE)["structural"],
                            candidate_signature(OP_CHANGED)["structural"])

    def test_extra_loop_is_a_different_candidate(self):
        self.assertNotEqual(candidate_signature(BASE)["structural"],
                            candidate_signature(LOOP_ADDED)["structural"])

    def test_two_different_real_kernels_are_not_collapsed(self):
        layernorm = '''import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, w, b, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) * (x - mean), axis=0) / N
    y = (x - mean) / tl.sqrt(var + 1e-5) * w + b
    tl.store(Y + row * N + cols, y)
'''
        self.assertNotEqual(candidate_signature(BASE)["structural"],
                            candidate_signature(layernorm)["structural"])


class TestDescriptor(unittest.TestCase):
    """描述符是**观测用**的: 它故意会把不同实现归到同一桶, 所以不参与判决"""

    def test_kernel_body_is_inside_the_key(self):
        """把这一条单独钉住: 第一版的 normalizer 在函数节点上 return 得太早, 于是**函数体
        完全不在哈希里**, 两份体不同而名字相同的核会撞成同一个键 —— 而「改名应当相等」那条
        用例照样绿, 因为它是通过对称性而不是通过差异来确认的。"""
        left = "def kernel(A, B):\n    return A @ B\n"
        right = "def kernel(A, B):\n    return (A @ B).T @ A\n"
        self.assertNotEqual(candidate_signature(left)["structural"],
                            candidate_signature(right)["structural"])

    def test_descriptor_names_the_apis_that_matter(self):
        sig = candidate_signature(BASE)
        apis, loops, kernels, cfg_items = sig["descriptor"]
        self.assertIn("load", apis)
        self.assertIn("store", apis)
        self.assertIn("sum", apis)
        self.assertEqual(loops, 1)
        self.assertEqual({"num_warps": 4, "num_stages": 2,
                          "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
                         dict(cfg_items),
                         "launch 配置没进描述符 —— 「配方换了核没变」就看不见")

    def test_bare_block_name_is_a_config(self):
        """标定: 真实生成代码里最常见的写法就是裸 `BLOCK`。第一版枚举名字时漏了它,
        于是「把 BLOCK 从 64 改成 128」这一刀在描述符里完全 invisible —— 而枚举表看起来
        已经覆盖了 BLOCK_M/BLOCK_N/BLOCK_K, 不会有人去查漏掉的那一个。"""
        bare = ("import triton\nimport triton.language as tl\n\n"
                "@triton.jit\n"
                "def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):\n"
                "    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
                "    tl.store(Y + o, tl.load(X + o), mask=o < N)\n"
                "def run(X, Y, N):\n"
                "    return kernel[(N // 1024,)](X, Y, N, BLOCK=1024, num_warps=8)\n")
        cfg = candidate_signature(bare)["configs"]
        self.assertEqual({"BLOCK": 1024, "num_warps": 8}, cfg)

    def test_config_change_shows_up_in_the_descriptor(self):
        self.assertNotEqual(candidate_signature(BASE)["descriptor"],
                            candidate_signature(BLOCK_CHANGED)["descriptor"],
                            "换了 BLOCK 却在描述符里看不出来 —— 那「配方换了核没变」就看不见")

    def test_runtime_dependent_config_is_not_invented(self):
        """BLOCK=next_power_of_2(N) 的值取决于运行期 shape, 不该被当成一个静态配置值"""
        dyn = BASE.replace("BLOCK_M=64", "BLOCK_M=triton.next_power_of_2(1024)")
        cfg = candidate_signature(dyn)["configs"]
        self.assertNotIn("BLOCK_M", cfg, "把运行期表达式记成静态配置 = 凭空造一个特征")


class TestUnparseable(unittest.TestCase):
    def test_syntax_error_returns_none_not_a_signature(self):
        """语法错的候选走不到 L3; 给它编一个键只会让重复率统计里混进无意义的命中"""
        self.assertIsNone(candidate_signature("def kernel(A, B:"))
        self.assertEqual("", signature_of("def kernel(A, B:"))
        self.assertIsNone(candidate_signature(""))

    def test_code_without_any_function_returns_none(self):
        """只有 import 的半个产出不是「一个候选」; 给它一个键会把一批失败读成同一批重复"""
        self.assertIsNone(candidate_signature("import triton\nimport triton.language as tl\n"))

    def test_signature_of_returns_the_structural_key(self):
        self.assertEqual(candidate_signature(BASE)["structural"], signature_of(BASE))


if __name__ == "__main__":
    unittest.main()
