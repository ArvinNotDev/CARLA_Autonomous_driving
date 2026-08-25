import time
import unittest
import numpy as np

from trajectory.worker import TrajectoryResult


class TestTrajectoryResultRetention(unittest.TestCase):
    def test_valid_prediction_remains_a_value_after_time_passes(self):
        pred = np.zeros((5, 2), dtype=np.float32)
        pred.setflags(write=False)
        result = TrajectoryResult(
            request_id=7,
            source_frame_id=100,
            command="LANE_FOLLOW",
            submitted_at=1.0,
            completed_at=time.perf_counter() - 2.0,
            steer=0.15,
            prediction=pred,
            valid=True,
        )
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.prediction)
        self.assertEqual(result.steer, 0.15)


if __name__ == "__main__":
    unittest.main()
