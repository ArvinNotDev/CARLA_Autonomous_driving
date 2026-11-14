from navigation.junction_trigger import JunctionTriggerLatch


def test_early_roi_detection_stays_pending_after_planner_rejects_it_as_too_far():
    latch = JunctionTriggerLatch(clear_duration=1.0)

    assert latch.observe(
        detected=True,
        evidence_present=True,
        timestamp=0.0,
        manager_idle=True,
    )

    # Planning returned to IDLE because the vehicle was still outside the
    # entry distance. The marker is still visible, so the event must survive.
    assert latch.observe(
        detected=False,
        evidence_present=True,
        timestamp=0.4,
        manager_idle=True,
    )
    assert latch.pending
    assert latch.armed


def test_model_detection_is_latched_until_a_sustained_negative_observation():
    latch = JunctionTriggerLatch(clear_duration=1.0)

    assert latch.observe(
        detected=True,
        evidence_present=True,
        timestamp=0.0,
        manager_idle=True,
    )
    assert latch.observe(
        detected=False,
        evidence_present=False,
        timestamp=0.4,
        manager_idle=True,
    )
    assert latch.pending

    # The negative model output has now persisted for the whole clear window.
    assert not latch.observe(
        detected=False,
        evidence_present=False,
        timestamp=1.5,
        manager_idle=True,
    )
    assert not latch.pending
    assert latch.armed


def test_successful_plan_disarms_repeated_crosswalk_detection_until_clear():
    latch = JunctionTriggerLatch(clear_duration=1.0)
    latch.observe(
        detected=True,
        evidence_present=True,
        timestamp=0.0,
        manager_idle=True,
    )

    latch.acknowledge()
    assert not latch.pending
    assert not latch.armed

    # A still-visible zebra stripe cannot trigger a second stop.
    assert not latch.observe(
        detected=True,
        evidence_present=True,
        timestamp=0.2,
        manager_idle=False,
    )

    # The marker must clear, and the junction sequence must be finished.
    assert not latch.observe(
        detected=False,
        evidence_present=False,
        timestamp=0.5,
        manager_idle=False,
    )
    assert not latch.observe(
        detected=False,
        evidence_present=False,
        timestamp=1.6,
        manager_idle=True,
    )
    assert latch.armed
    assert not latch.pending

    assert latch.observe(
        detected=True,
        evidence_present=True,
        timestamp=1.7,
        manager_idle=True,
    )
