def test_package_imports_and_has_version():
    import aslv2
    assert isinstance(aslv2.__version__, str)
    assert aslv2.__version__ != ""
