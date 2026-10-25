import qec_diverse_circuits


def test_version_is_exposed_and_nonempty():
    assert hasattr(qec_diverse_circuits, "__version__")
    assert isinstance(qec_diverse_circuits.__version__, str)
    assert len(qec_diverse_circuits.__version__) > 0
