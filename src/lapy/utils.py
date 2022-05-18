def test_skparse() -> bool:
    try:
        from sksparse.cholmod import cholesky  # noqa: F401
    except ImportError:
        return False
    else:
        return True
