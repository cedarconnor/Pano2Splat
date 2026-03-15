"""Quick test: verify gsplat CUDA backend compiles and loads."""


def test_gsplat_cuda_backend_loads() -> None:
    from gsplat.cuda._backend import _C

    print("gsplat CUDA backend OK:", _C)
    ops = [x for x in dir(_C) if not x.startswith("_")]
    print(f"Available ops ({len(ops)}):", ops[:10], "...")
    assert ops
