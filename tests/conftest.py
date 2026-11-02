"""Keep application side effects out of the offline test process."""

import os


# Some test modules import ``main`` during collection.  Set the guard here,
# before pytest imports any of them, so collection cannot start the scheduler,
# webhook registration, Telegram notifications, or the immediate scan thread.
os.environ["BOT_STARTUP_ENABLED"] = "0"
