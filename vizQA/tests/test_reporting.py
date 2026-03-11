import unittest
from unittest.mock import MagicMock, patch


class TestFailureReporting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from vizQA.core import Automator

        self.mock_client = MagicMock()
        self.automator = Automator(self.mock_client, verbosity=2)

    def test_get_failure_details_v0(self):
        self.automator.verbosity = 0
        perception = {"elements": [{"text": "Login"}], "top_matches": []}
        reason = self.automator._failure_details("VERIFY", "Submit", perception, "Verification failed")
        self.assertEqual(reason, "Verification failed for query: 'Submit'")

    def test_get_failure_details_v1(self):
        self.automator.verbosity = 1
        perception = {"elements": [{"text": "Login"}, {"text": "Cancel"}], "top_matches": []}
        reason = self.automator._failure_details("VERIFY", "Submit", perception, "Verification failed")
        self.assertIn("Elements visible on screen:", reason)
        self.assertIn("'Login'", reason)
        self.assertIn("'Cancel'", reason)

    async def test_failure_propagation(self):
        from vizQA.memory import FailureType, StepStatus, TestSession, TestStep

        # Create a parent step with a failed sub-step
        sub_step = TestStep(id="sub1", instruction="VERIFY: admin")
        sub_step.status = StepStatus.FAILED
        sub_step.failure_type = FailureType.PERCEPTION_MISMATCH
        sub_step.failure_reason = "Visual mismatch detected"

        parent_step = TestStep(id="parent1", instruction="Container step", sub_steps=[sub_step])

        # Define session
        session = TestSession(id="test_sess", test_name="Test", url="http://test.com")

        # Mock the run_step_recursive to return False for the sub_step
        # but execute the real logic for the parent step.
        real_run = self.automator._run_step_recursive

        async def mock_run_recursive(sess, stp, upd):
            if stp.id == "parent1":
                return await real_run(sess, stp, upd)
            stp.status = StepStatus.FAILED  # Simulate child failure
            return False

        with patch.object(self.automator, "_run_step_recursive", side_effect=mock_run_recursive):
            success = await self.automator._run_step_recursive(session, parent_step, None)
            self.assertFalse(success)
            self.assertEqual(parent_step.status, StepStatus.FAILED)
            self.assertEqual(parent_step.failure_reason, "Visual mismatch detected")

    def test_get_failure_details_v2(self):
        self.automator.verbosity = 2
        perception = {"elements": [{"text": "Login"}], "top_matches": [{"text": "Subtitle", "similarity": 0.45}]}
        reason = self.automator._failure_details("VERIFY", "Submit", perception, "Verification failed")
        self.assertIn("Top candidates: 'Subtitle' (similarity: 0.45)", reason)
