from activation_report import choose_device

def test_explicit_cpu_device_is_preserved():
    assert choose_device("cpu") == "cpu"
