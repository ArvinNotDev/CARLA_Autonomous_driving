import time
import unittest
import numpy as np

from trajectory.worker import InferenceWorker


class FakeTrajectoryAgent:
    def __init__(self):
        self.prev_steer = 0.0
        self._worker_command = None

    def get_steering_and_pred(self, frame, command_name=None):
        # The worker must serialize these calls and publish immutable results.
        time.sleep(0.01)
        value = float(frame[0, 0, 0])
        return value, np.array([[value, 0.0]], dtype=np.float32)


class TestWorkerOrdering(unittest.TestCase):
    def test_newer_request_replaces_pending_request(self):
        worker = InferenceWorker(trajectory_agent=FakeTrajectoryAgent())
        worker.start()
        try:
            frame1 = np.zeros((2, 2, 3), dtype=np.uint8)
            frame2 = np.full((2, 2, 3), 2, dtype=np.uint8)
            frame3 = np.full((2, 2, 3), 3, dtype=np.uint8)

            worker.submit_trajectory(source_frame_id=1, bgr_frame=frame1, command="LANE_FOLLOW")
            worker.submit_trajectory(source_frame_id=2, bgr_frame=frame2, command="LANE_FOLLOW")
            latest_id = worker.submit_trajectory(source_frame_id=3, bgr_frame=frame3, command="LANE_FOLLOW")

            deadline = time.time() + 1.0
            result = None
            while time.time() < deadline:
                result, _, _, _ = worker.trajectory_snapshot()
                if result is not None and result.request_id == latest_id:
                    break
                time.sleep(0.01)

            self.assertIsNotNone(result)
            self.assertEqual(result.request_id, latest_id)
            self.assertEqual(result.source_frame_id, 3)
            self.assertEqual(float(result.steer), 3.0)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
