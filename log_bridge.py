import sys
from PyQt5.QtCore import QObject, pyqtSignal

'''
Allows for print statements to be redirected from other files to the log panel in the UI.
Any print() calls will automatically be displayed on the log panel in the app.
'''

class LogBridge(QObject):
    message_received = pyqtSignal(str)

    def write(self, text: str):
        if text and text.strip():
            self.message_received.emit(text.rstrip())

    def flush(self):
        pass

bridge = LogBridge()