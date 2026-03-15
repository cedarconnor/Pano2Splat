"""Verify all key libraries import correctly."""


def test_all_imports() -> None:
    import flash_attn
    import sageattention
    import torch
    import triton
    import xformers
    from gsplat.cuda._backend import _C

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"xformers {xformers.__version__}")
    print(f"triton {triton.__version__}")
    print(f"flash_attn {flash_attn.__version__}")

    try:
        print(f"sageattention {sageattention.__version__}")
    except AttributeError:
        print("sageattention OK (no __version__)")

    ops = [x for x in dir(_C) if not x.startswith("_")]
    print(f"gsplat CUDA: OK ({len(ops)} ops)")
    assert ops
