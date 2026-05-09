import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFailureReporting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from vizQA.app.core import Automator

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
        from vizQA.app.memory import FailureType, StepStatus, TestSession, TestStep

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

    async def test_execute_find_logs_selected_element_only_at_highest_verbosity(self):
        from vizQA.app.memory import TestSession, TestStep

        session = TestSession(id="test_sess", test_name="Test", url="http://test.com")
        step = TestStep(id="find1", instruction="FIND: Sign in")
        perception = {"elements": [{"text": "Sign in"}]}

        self.automator.page = MagicMock()
        self.automator.page.screenshot = AsyncMock()
        self.automator.client.perceive = AsyncMock(return_value=perception)
        self.automator.logger = MagicMock()

        self.automator.verbosity = 1
        success = await self.automator._execute_find(session, step, "Sign in")
        self.assertTrue(success)
        self.automator.logger.log_perception.assert_not_called()

        self.automator.verbosity = 2
        self.automator.logger.reset_mock()
        success = await self.automator._execute_find(session, step, "Sign in")
        self.assertTrue(success)
        self.automator.logger.log_perception.assert_called_once_with(
            step.id, "Sign in", perception, selected=perception["elements"][0]
        )

    async def test_execute_verify_logs_candidates_with_no_selected_element(self):
        from vizQA.app.memory import TestSession, TestStep

        session = TestSession(id="test_sess", test_name="Test", url="http://test.com")
        step = TestStep(id="verify1", instruction="VERIFY: success")
        perception = {"elements": [{"text": "Success banner"}]}

        self.automator.page = MagicMock()
        self.automator.page.screenshot = AsyncMock()
        self.automator.client.perceive = AsyncMock(return_value=perception)
        self.automator.logger = MagicMock()
        self.automator.verbosity = 2
        self.automator.parser.parse_verify_intent = MagicMock(
            return_value={"keyword": "success", "subject": "", "position": "", "color": None, "negated": False}
        )

        with (
            patch.object(self.automator, "_check_verification_match", return_value=(True, "")),
            patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock),
        ):
            success = await self.automator._execute_verify(session, step, "success", timeout=1)

        self.assertTrue(success)
        self.automator.logger.log_perception.assert_called_once_with(step.id, "'success'  ", perception, selected=None)
