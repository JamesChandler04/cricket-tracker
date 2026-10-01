import sys
from PyQt5.QtCore import QObject, pyqtSignal

class LogBridge(QObject):
    message_received = pyqtSignal(str)

    def write(self, text: str):
        if text and text.strip():
            self.message_received.emit(text.rstrip())

    def flush(self):
        pass

bridge = LogBridge()