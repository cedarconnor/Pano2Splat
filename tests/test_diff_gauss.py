def test_diff_gaussian_rasterization_import() -> None:
    from diff_gaussian_rasterization import _C

    print("diff_gaussian_rasterization OK")
    assert _C is not None
