import unittest
from datetime import datetime,timezone
from cash_session_lab import session
class CashSessionTests(unittest.TestCase):
    def at(self,text):return session(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())
    def test_dst_changes_utc_open(self):
        self.assertEqual(self.at('2026-03-06T14:30:00'),'first_hour')
        self.assertEqual(self.at('2026-03-09T13:30:00'),'first_hour')
        self.assertEqual(self.at('2026-03-06T13:30:00'),'outside')
    def test_holiday_and_early_close(self):
        self.assertEqual(self.at('2026-07-03T15:00:00'),'closed')
        self.assertEqual(self.at('2026-11-27T18:00:00'),'outside')
        self.assertEqual(self.at('2026-11-27T17:30:00'),'last_hour')
