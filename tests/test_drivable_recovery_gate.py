from driving.recovery_gate import ContinuousConditionGate


def test_condition_must_remain_true_for_full_confirmation_period():
    gate = ContinuousConditionGate()

    assert not gate.update(True, timestamp=10.0, required_seconds=1.5)
    assert not gate.update(True, timestamp=11.49, required_seconds=1.5)
    assert gate.update(True, timestamp=11.5, required_seconds=1.5)


def test_brief_error_drop_resets_confirmation_timer():
    gate = ContinuousConditionGate()

    assert not gate.update(True, timestamp=10.0, required_seconds=1.5)
    assert not gate.update(True, timestamp=11.0, required_seconds=1.5)
    assert not gate.update(False, timestamp=11.1, required_seconds=1.5)

    # A new error episode must complete its own full confirmation period.
    assert not gate.update(True, timestamp=11.2, required_seconds=1.5)
    assert not gate.update(True, timestamp=12.69, required_seconds=1.5)
    assert gate.update(True, timestamp=12.7, required_seconds=1.5)


def test_reset_rearms_gate():
    gate = ContinuousConditionGate()

    assert not gate.update(True, timestamp=1.0, required_seconds=0.5)
    gate.reset()
    assert not gate.update(True, timestamp=5.0, required_seconds=0.5)
    assert gate.update(True, timestamp=5.5, required_seconds=0.5)


def test_zero_confirmation_duration_is_immediate():
    gate = ContinuousConditionGate()

    assert gate.update(True, timestamp=3.0, required_seconds=0.0)
    assert not gate.update(False, timestamp=3.1, required_seconds=0.0)
