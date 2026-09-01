import unittest
from unittest.mock import AsyncMock, patch

from discord_mistral_bot import parse_schedule_intent, validate_schedule_intent


class ValidateScheduleIntentTests(unittest.TestCase):
    def test_add_requires_tickers_and_time(self):
        self.assertEqual(
            validate_schedule_intent(
                {"action": "add", "tickers": ["AAPL", "ORSTED.CPH"], "time": "08:00"}
            ),
            {"action": "add", "tickers": ["AAPL", "ORSTED.CPH"], "time": "08:00"},
        )
        self.assertEqual(validate_schedule_intent({"action": "add"}), {"action": "unknown"})

    def test_remove_requires_hex_job_id(self):
        self.assertEqual(
            validate_schedule_intent({"action": "remove", "id": "A1B2C3D4"}),
            {"action": "remove", "id": "a1b2c3d4"},
        )
        self.assertEqual(
            validate_schedule_intent({"action": "remove", "id": "delete-all"}),
            {"action": "unknown"},
        )

    def test_edit_requires_job_id_and_tickers(self):
        self.assertEqual(
            validate_schedule_intent(
                {"action": "edit", "id": "1234abcd", "tickers": ["MSFT"]}
            ),
            {"action": "edit", "id": "1234abcd", "tickers": ["MSFT"]},
        )
        self.assertEqual(
            validate_schedule_intent({"action": "edit", "id": "1234abcd", "tickers": []}),
            {"action": "unknown"},
        )

    def test_unknown_or_malformed_input_is_safe(self):
        for value in (None, [], "add AAPL", {}, {"action": "anything"}):
            self.assertEqual(validate_schedule_intent(value), {"action": "unknown"})


class ParseScheduleIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_parser_requests_deterministic_schema_output(self):
        with patch(
            "discord_mistral_bot.asyncio.to_thread",
            new=AsyncMock(return_value='{"action":"add","tickers":["AAPL"],"time":"08:00"}'),
        ) as to_thread:
            result = await parse_schedule_intent("add Apple at 8", "mistral")

        self.assertEqual(
            result,
            {"action": "add", "tickers": ["AAPL"], "time": "08:00"},
        )
        self.assertEqual(to_thread.await_args.kwargs["temperature"], 0)
        self.assertIsInstance(to_thread.await_args.kwargs["response_format"], dict)


if __name__ == "__main__":
    unittest.main()
