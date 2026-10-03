"""Sends printed text to the GUI's log panel."""

from __future__ import annotations

import sys
from PyQt5.QtCore import QObject, pyqtSignal

class LogBridge(QObject):
    """Stand-in for sys.stdout that passes printed text to the GUI as a Qt signal."""
    message_received = pyqtSignal(str)

    def write(self, text: str) -> None:
        """Send non-blank text to the log as one message."""
        if text and text.strip():
            self.message_received.emit(text.rstrip())

    def flush(self) -> None:
        """Do nothing; sys.stdout replacements must have this method."""
        pass

bridge = LogBridge()
"""The LogBridge the GUIs swap in for sys.stdout."""