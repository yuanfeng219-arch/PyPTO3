@pl.jit.incore
def mm(
    a: pl.Tensor[[32, 32], pl.FP16],
    b: pl.Tensor[[32, 32], pl.FP16],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)
    c_acc = pl.matmul(a_l0a, b_l0b)      # 落在 Acc
    pl.store(c_acc, [0, 0], out)         # Acc -> DDR
    return out
