"""
Active-control NetHack RL GUI

This version hooks directly into an NethackController instance,
overwriting the previous queue/thread approach.

Each call to controller.step(...) now always returns a StepInfo,
which embeds the game frame (chars+colors), action, reward, labels,
and optional ending. The frame and color arrays are now guaranteed
to be the same shape (24 rows x 80 columns).
"""
from __future__ import annotations
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# pylint: disable=c-extension-no-member,invalid-name

# --------------------------------------------------------------------------- #
#                          Data‑passing primitives                            #
# --------------------------------------------------------------------------- #

@dataclass
class TerminalFrame:
    """
    A full NetHack terminal frame:
      - chars: 24 strings of exactly 80 characters each
      - colors: 24x80 integers (ANSI 0-15) matching chars
    """
    chars: List[str]
    colors: List[List[int]]

@dataclass
class StepInfo:
    """Information about a single step in the game.
    Contains the current frame, action taken, reward received,
    labels for rewards, and an optional ending."""
    frame: TerminalFrame
    action: str                    # e.g. 'NE' or '' for no-op
    reward: float
    labels: List[Tuple[str, float]]
    ending: Optional[str] = None

# --------------------------------------------------------------------------- #
#                            External controller                             #
# --------------------------------------------------------------------------- #

class NethackController:
    """
    Your NetHack interface must implement:
      - reset() -> TerminalFrame
      - step(action: Optional[int]) -> StepInfo
    """
    def reset(self) -> TerminalFrame:
        """Reset the simulation."""
        raise NotImplementedError

    def step(self, action: Optional[int] = None) -> StepInfo:
        """Take a step in the simulation."""
        raise NotImplementedError

# --------------------------------------------------------------------------- #
#                               Terminal widget                               #
# --------------------------------------------------------------------------- #

class TerminalWidget(QtWidgets.QWidget):
    """Widget displaying a fixed 24×80 text grid with ANSI colors."""
    def __init__(self, rows: int = 24, cols: int = 80, parent=None) -> None:
        super().__init__(parent)
        self.rows, self.cols = rows, cols
        # initialize blank frame
        self.chars = [" " * cols for _ in range(rows)]
        self.colors = [[7] * cols for _ in range(rows)]
        # monospace font setup
        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setFixedPitch(True)
        self.setFont(font)
        fm = QtGui.QFontMetrics(font)
        self.char_w = fm.horizontalAdvance('W')
        self.char_h = fm.height()
        self.setMinimumSize(self.char_w * cols, self.char_h * rows)

    def set_frame(self, frame: TerminalFrame) -> None:
        """Set the terminal frame to display."""
        # enforce exact 24×80 shape
        assert len(frame.chars) == self.rows, "Frame rows mismatch"
        assert all(len(line) == self.cols for line in frame.chars), "Char line length mismatch"
        assert len(frame.colors) == self.rows, "Color rows mismatch"
        assert all(len(row) == self.cols for row in frame.colors), "Color row length mismatch"
        # copy data
        self.chars = frame.chars.copy()
        self.colors = [row.copy() for row in frame.colors]
        self.update()

    def paintEvent(self, _) -> None:
        """Paint the terminal widget."""
        painter = QtGui.QPainter(self)
        painter.setFont(self.font())

        painter.fillRect(0, 0, self.char_w * self.cols,
                         self.char_h * self.rows,
                         QtGui.QColor(0, 0, 0))
        # ANSI 16-color palette
        ansi: List[QtGui.QColor] = [QtGui.QColor(r, g, b) for r, g, b in [
            (0,0,0),(128,0,0),(0,128,0),(128,128,0),
            (0,0,128),(128,0,128),(0,128,128),(192,192,192),
            (128,128,128),(255,0,0),(0,255,0),(255,255,0),
            (0,0,255),(255,0,255),(0,255,255),(255,255,255)
        ]]
        for r in range(self.rows):
            y = (r + 1) * self.char_h
            row_chars = self.chars[r]
            row_colors = self.colors[r]
            for c in range(self.cols):
                painter.setPen(ansi[row_colors[c]])
                # coerce uint8 / bytes to a one‐char str
                val = row_chars[c]
                if isinstance(val, str):
                    ch = val
                else:
                    # handles numpy.uint8 or bytes
                    ch = chr(int(val))
                painter.drawText(c * self.char_w, y, ch)
        painter.end()

# --------------------------------------------------------------------------- #
#                                  The  UI                                    #
# --------------------------------------------------------------------------- #

class NetHackWindow(QtWidgets.QMainWindow):
    """Main window for the NetHack debugger."""
    def __init__(self, controller: NethackController) -> None:
        super().__init__()
        self.controller = controller
        self.paused = True
        self._build_ui()
        self._init_run()

    def _build_ui(self) -> None:
        self.setWindowTitle("NetHack RL GUI")
        self.resize(1000, 800)
        # Top button bar (fixed height)
        buttons = [("Restart", self._on_restart),
                   ("Play ▷", self._on_play_pause),
                   ("Step ➔", self._on_step)]
        top_layout = QtWidgets.QHBoxLayout()
        for label, handler in buttons:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(handler)
            top_layout.addWidget(btn)
        top_layout.addStretch()
        top_widget = QtWidgets.QWidget()
        top_widget.setLayout(top_layout)
        top_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Fixed)
        # Main split: terminal (left) + actions (right)
        self.terminal = TerminalWidget()
        self.actions = QtWidgets.QTreeWidget()
        self.actions.setHeaderLabels(["Action", "Reward"])
        hdr = self.actions.header()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        mid_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        mid_split.addWidget(self.terminal)
        mid_split.addWidget(self.actions)
        mid_split.setStretchFactor(0, 2)
        mid_split.setStretchFactor(1, 3)
        # Bottom: rewards summary | endings
        self.rewards = self._make_table("Reward label", "Total")
        self.endings = self._make_table("Ending", "Count")
        bot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot_split.addWidget(self.rewards)
        bot_split.addWidget(self.endings)
        # Compose layout
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(top_widget)
        v_layout.addWidget(mid_split, 1)
        v_layout.addWidget(bot_split, 0)
        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)
        # Timer for auto-stepping
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_step)
        self._rewards_counter: defaultdict[str, float] = defaultdict(float)

    def _init_run(self) -> None:
        frame = self.controller.reset()
        self.terminal.set_frame(frame)
        self.actions.clear()
        self._rewards_counter.clear()
        self._refresh_rewards()

    def _on_restart(self) -> None:
        self._timer.stop()
        self.paused = True
        self._init_run()

    def _on_play_pause(self) -> None:
        self.paused = not self.paused
        btn = self.sender()
        if isinstance(btn, QtWidgets.QPushButton):
            btn.setText("Pause ⏸" if not self.paused else "Play ▷")
        if self.paused:
            self._timer.stop()
        else:
            self._timer.start(100)

    def _on_step(self, action: Optional[int] = None) -> None:
        act = action if isinstance(action, int) and not isinstance(action, bool) else None
        result = self.controller.step(act)
        if result is not None:
            assert isinstance(result, StepInfo), f"Expected StepInfo, got {type(result)}"
            self.terminal.set_frame(result.frame)
            self._add_step(result)
            if result.ending is not None:
                self._finish_episode(result.ending.name)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key presses for actions."""
        key = event.key()
        modifiers = event.modifiers()

        numpad_mapping = {
            QtCore.Qt.Key_7: 'y',
            QtCore.Qt.Key_8: 'k',
            QtCore.Qt.Key_9: 'u',
            QtCore.Qt.Key_4: 'h',
            QtCore.Qt.Key_6: 'l',
            QtCore.Qt.Key_1: 'b',
            QtCore.Qt.Key_2: 'j',
            QtCore.Qt.Key_3: 'n',
        }

        if modifiers & QtCore.Qt.KeypadModifier and key in numpad_mapping:
            code = ord(numpad_mapping[key])
        elif txt := event.text():
            code = ord(txt)
        elif (modifiers & QtCore.Qt.ControlModifier and
            QtCore.Qt.Key_A <= key <= QtCore.Qt.Key_Z):
            code = (key - QtCore.Qt.Key_A) + 1
        else:
            return

        self._on_step(code)

    def _add_step(self, step: StepInfo) -> None:
        item = QtWidgets.QTreeWidgetItem([step.action, f"{step.reward:+.2f}"])
        for lbl, val in step.labels:
            QtWidgets.QTreeWidgetItem(item, [f"• {lbl}", f"{val:+.2f}"])
            self._rewards_counter[lbl] += val
        self.actions.addTopLevelItem(item)
        self.actions.scrollToItem(item)
        self._refresh_rewards()

    def _finish_episode(self, ending: str) -> None:
        cnt = Counter({
            self.endings.item(r, 0).text(): int(self.endings.item(r, 1).text())
            for r in range(self.endings.rowCount())
        })
        cnt[ending] += 1
        self._populate(self.endings, cnt.most_common())
        self.actions.clear()
        self._rewards_counter.clear()
        self._refresh_rewards()
        self._init_run()

    def _refresh_rewards(self) -> None:
        items = sorted(self._rewards_counter.items(), key=lambda kv: kv[1], reverse=True)
        self._populate(self.rewards, items)

    def _make_table(self, c0: str, c1: str) -> QtWidgets.QTableWidget:
        tbl = QtWidgets.QTableWidget(0, 2)
        tbl.setHorizontalHeaderLabels([c0, c1])
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        return tbl

    def _populate(self, table: QtWidgets.QTableWidget, rows: List[Tuple[str, float | int]]) -> None:
        table.setRowCount(len(rows))
        for r, (lbl, val) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(lbl))
            tbl_txt = f"{val:+.2f}" if isinstance(val, float) else str(val)
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(tbl_txt))
        table.resizeRowsToContents()

# --------------------------------------------------------------------------- #
#                                   Entrypoint                               #
# --------------------------------------------------------------------------- #

def run_gui(controller: NethackController) -> None:
    """Run the NetHack GUI debugger with the given controller."""
    app = QtWidgets.QApplication(sys.argv)
    win = NetHackWindow(controller)
    win.show()
    sys.exit(app.exec())

# --------------------------------------------------------------------------- #
#                         Example stub controller                           #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    class DemoCtrl(NethackController):
        """Demo to ensure the UI renders."""
        def __init__(self):
            self.steps = 0

        def reset(self) -> TerminalFrame:
            self.steps = 0
            return TerminalFrame(chars=['.'*80]*24, colors=[[7]*80 for _ in range(24)])

        def step(self, action: Optional[int] = None) -> StepInfo:
            time.sleep(0.05)
            self.steps += 1
            pos = self.steps % 80
            chars = [ ''.join('@' if col==pos else '.' for col in range(80)) for _ in range(24) ]
            colors = [[7]*80 for _ in range(24)]
            frame = TerminalFrame(chars=chars, colors=colors)
            if self.steps < 50:
                return StepInfo(frame, 'S', 0.0, [])
            if self.steps < 55:
                return StepInfo(frame, 'N', 0.1, [('test',0.1)])
            return StepInfo(frame, 'S', 1.0, [('test',1.0)], ending='DemoEnd')

    run_gui(DemoCtrl())
