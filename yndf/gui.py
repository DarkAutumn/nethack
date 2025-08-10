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
from enum import Enum
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from yndf.nethack_state import NethackState
from yndf.movement import UNPASSABLE_WAVEFRONT, GlyphKind

# pylint: disable=c-extension-no-member,invalid-name

# --------------------------------------------------------------------------- #
#                          Data‑passing primitives                            #
# --------------------------------------------------------------------------- #

@dataclass
class StepInfo:
    """Information about a single step in the game."""
    state: NethackState
    action: str
    reward: float
    reward_labels: List[Tuple[str, float]]
    properties : Dict[str, object]
    ending: Optional[Enum | str] = None

# --------------------------------------------------------------------------- #
#                            External controller                             #
# --------------------------------------------------------------------------- #

class NethackController:
    """
    Your NetHack interface must implement:
      - reset() -> TerminalFrame
      - step(action: Optional[int]) -> StepInfo
    """
    def reset(self) -> NethackState:
        """Reset the simulation."""
        raise NotImplementedError

    def step(self, action: Optional[int] = None) -> StepInfo:
        """Take a step in the simulation."""
        raise NotImplementedError

    def set_model(self, model_path: str) -> None:
        """Set the current model path for the controller."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#                               Terminal widget                               #
# --------------------------------------------------------------------------- #

class TerminalWidget(QtWidgets.QWidget):
    """Widget displaying a fixed 24×80 text grid with ANSI colors."""
    def __init__(self, rows: int = 24, cols: int = 80, parent=None) -> None:
        super().__init__(parent)
        self.rows, self.cols = rows, cols
        # initialize state
        self.state : NethackState = None
        # monospace font setup
        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setFixedPitch(True)
        self.setFont(font)
        fm = QtGui.QFontMetrics(font)
        self.char_w = fm.horizontalAdvance('W')
        self.char_h = fm.height()
        self.setMinimumSize(self.char_w * cols, self.char_h * rows)

        self.setMouseTracking(True)
        self._last_hover_cell: Tuple[int, int] = (-1, -1)

    def set_frame(self, state : NethackState) -> None:
        """Set the terminal frame to display."""
        # enforce exact 24×80 shape
        assert len(state.tty_chars) == self.rows, "Frame rows mismatch"
        assert all(len(line) == self.cols for line in state.tty_chars), "Char line length mismatch"
        assert len(state.tty_colors) == self.rows, "Color rows mismatch"
        assert all(len(row) == self.cols for row in state.tty_colors), "Color row length mismatch"
        # copy data
        self.state = state
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement to show tooltips with cell info."""
        # In Qt6, positions are QPointF
        pos = event.position()
        col = int(pos.x() // self.char_w)
        row = int(pos.y() // self.char_h)

        if 0 <= col < self.cols and 0 <= row < self.rows:
            if (col, row) != self._last_hover_cell:
                self._last_hover_cell = (col, row)
                text = self.getHoverText(col, row)  # (x, y) == (col, row)
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(),  # screen coords
                    text,
                    self
                )
        else:
            # outside the grid
            QtWidgets.QToolTip.hideText()
            self._last_hover_cell = (-1, -1)

        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle mouse leaving the widget area."""
        QtWidgets.QToolTip.hideText()
        self._last_hover_cell = (-1, -1)
        super().leaveEvent(event)

    def getHoverText(self, x: int, y: int) -> str:
        """Get the text at the given coordinates, handling the 21x79 glyph map."""
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return ""

        ch = self.state.tty_chars[y][x]
        color = self.state.tty_colors[y][x]

        # NetHack tty defaults: map is 21 rows x 79 cols starting at (row 1, col 0)
        MAP_ROW0 = 1
        MAP_COL0 = 0

        gy = y - MAP_ROW0
        gx = x - MAP_COL0
        if not (0 <= gy < self.state.glyphs.shape[0] and 0 <= gx < self.state.glyphs.shape[1]):
            return f"Pos: ({gy}, {gx})\nGlyph: Out of bounds"

        tooltip = []

        tooltip.append(self.state.get_screen_description((gy, gx)))
        tooltip.append(f"Pos: ({gy}, {gx})")
        tooltip.append(f"Glyph: {str(self.state.glyphs[gy][gx])}, Char: {ch}, Color: {color}")
        tooltip.append(f"Floor Glyph: {self.state.floor_glyphs[gy, gx]}")

        if 0 <= gy < self.state.glyph_kinds.shape[0] and 0 <= gx < self.state.glyph_kinds.shape[1]:
            kind_val = self.state.glyph_kinds[gy, gx]
            kind_val = GlyphKind(kind_val) if kind_val in GlyphKind else kind_val
            tooltip.append(f"Glyph Kind: {kind_val.name}")

        if 0 <= gy < self.state.wavefront.shape[0] and 0 <= gx < self.state.wavefront.shape[1]:
            wave_val = self.state.wavefront[gy, gx]
            if wave_val == UNPASSABLE_WAVEFRONT:
                tooltip.append("Wavefront: Unpassable")
            else:
                tooltip.append(f"Wavefront: {wave_val}")

        return "\n".join(tooltip)

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
            row_chars = self.state.tty_chars[r]
            row_colors = self.state.tty_colors[r]
            for c in range(self.cols):
                color_index = row_colors[c]
                if color_index < 0 or color_index >= len(ansi):
                    color = QtGui.QColor(238,130,238)
                else:
                    color = ansi[color_index]
                painter.setPen(color)
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
    def __init__(self, controller: NethackController, model_path : str) -> None:
        super().__init__()
        self.controller = controller
        self.model_path: Path | None = Path(model_path) if model_path else None
        self.paused = True
        self._build_ui()
        self._init_run()

    def _build_ui(self) -> None:
        self.setWindowTitle("NetHack RL GUI")
        self.resize(1400, 900)

        top_layout = QtWidgets.QHBoxLayout()

        self.model_select = QtWidgets.QComboBox()
        self.model_select.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_select.setToolTip("Select a model (.zip)")
        top_layout.addWidget(self.model_select)

        buttons = [("Restart", self._on_restart),
                   ("Play ▷", self._on_play_pause),
                   ("Step ➔", self._on_step)]
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

        self.actions.setMinimumWidth(320)

        mid_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        mid_split.addWidget(self.terminal)
        mid_split.addWidget(self.actions)
        mid_split.setStretchFactor(0, 3)
        mid_split.setStretchFactor(1, 2)
        # Bottom: rewards summary | endings
        self.rewards = self._make_table("Reward label", "Total")
        self.endings = self._make_table("Ending", "Count")
        bot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot_split.addWidget(self.rewards)
        bot_split.addWidget(self.endings)


        # Right: RunInfo table on the far right ---
        self.runinfo = self._make_table("Name", "Value")
        self.runinfo.verticalHeader().setVisible(False)
        self.runinfo.setMinimumWidth(300)

        # Left column stacks mid + bottom; right is RunInfo ---
        left_col = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(mid_split, 1)
        left_layout.addWidget(bot_split, 0)


        right_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        right_split.addWidget(left_col)
        right_split.addWidget(self.runinfo)
        right_split.setStretchFactor(0, 5)
        right_split.setStretchFactor(1, 2)


        term_w = self.terminal.char_w * self.terminal.cols
        actions_w = 360     # starting width for Actions/Rewards pane
        runinfo_w = 340     # starting width for RunInfo

        mid_split.setSizes([term_w, actions_w])
        right_split.setSizes([term_w + actions_w + 40, runinfo_w])  # +40 ≈ splitter/scrollbar slop


        # Compose everything
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(top_widget)
        v_layout.addWidget(right_split, 1)

        container = QtWidgets.QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        # Timer + latest status store
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_step)
        self._rewards_counter: defaultdict[str, float] = defaultdict(float)
        self._latest_status: dict[str, object] = {}

        self._populate_model_dropdown()
        self.model_select.currentTextChanged.connect(self._on_model_selected)

    def _init_run(self) -> None:
        frame = self.controller.reset()
        self.terminal.set_frame(frame)
        self.actions.clear()
        self._rewards_counter.clear()

        self._latest_status.clear()
        self._refresh_runinfo()
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
        result : StepInfo = self.controller.step(act)
        if result is not None:
            assert isinstance(result, StepInfo), f"Expected StepInfo, got {type(result)}"
            self.terminal.set_frame(result.state)
            self._add_step(result)
            if result.ending is not None:
                ending = result.ending.name if isinstance(result.ending, Enum) else str(result.ending)
                self._finish_episode(ending)

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
        for lbl, val in step.reward_labels:
            QtWidgets.QTreeWidgetItem(item, [f"• {lbl}", f"{val:+.2f}"])
            self._rewards_counter[lbl] += val
        self.actions.addTopLevelItem(item)
        self.actions.scrollToItem(item)

        status = step.state.as_dict().copy()
        status.update(step.properties or {})

        self._latest_status.update(status)
        self._refresh_runinfo()

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
        self._latest_status.clear()
        self._refresh_runinfo()
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

    def _populate_kv(self, table: QtWidgets.QTableWidget, rows: List[Tuple[str, str]]) -> None:
        table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
        table.resizeRowsToContents()

    def _refresh_runinfo(self) -> None:
        # show latest known status, sorted by key for stability
        items = [(k, self._fmt_val(v)) for k, v in sorted(self._latest_status.items(), key=lambda kv: kv[0])]
        self._populate_kv(self.runinfo, items)

    @staticmethod
    def _fmt_val(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def _populate_model_dropdown(self) -> None:  # NEW
        """Fill the model dropdown with .zip files from model_path."""
        if not hasattr(self, "model_select"):
            return
        self.model_select.blockSignals(True)
        self.model_select.clear()

        names: list[str] = []
        if self.model_path and self.model_path.exists():
            names = sorted([p.name for p in self.model_path.glob("*.zip")])

        self.model_select.addItems(names)
        self.model_select.setEnabled(bool(names))
        self.model_select.blockSignals(False)

        if names:
            # Default to the longest filename
            idx = max(range(len(names)), key=lambda i: len(names[i]))
            self.model_select.setCurrentIndex(idx)
            # Explicitly set model on initial selection
            self._on_model_selected(names[idx])

    def _on_model_selected(self, name: str) -> None:  # NEW
        """Handle model selection change from the dropdown."""
        if not name or not self.model_path:
            return
        full_path = str(self.model_path / name)
        if hasattr(self.controller, "set_model"):
            self.controller.set_model(full_path)

# --------------------------------------------------------------------------- #
#                                   Entrypoint                               #
# --------------------------------------------------------------------------- #

def run_gui(controller: NethackController, model_path: str) -> None:
    """Run the NetHack GUI debugger with the given controller."""
    app = QtWidgets.QApplication(sys.argv)
    win = NetHackWindow(controller, model_path)
    win.show()
    sys.exit(app.exec())

# --------------------------------------------------------------------------- #
#                         Example stub controller                           #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    class MockState:
        """Mock state for demo purposes."""
        def __init__(self):
            self.tty_chars = ['.'*80]*24
            self.tty_colors = [[7]*80 for _ in range(24)]
            self.glyphs = [[0]*79 for _ in range(21)]

        def as_dict(self):
            """Return a mock state dictionary."""
            return {
                "steps": 0,
                "pos": 0,
                "hp": 20,
                "dlvl": 1
            }

    class DemoCtrl(NethackController):
        """Demo to ensure the UI renders."""
        def __init__(self):
            self.steps = 0

        def reset(self):
            self.steps = 0
            return MockState()

        def step(self, _: Optional[int] = None) -> StepInfo:
            time.sleep(0.05)
            self.steps += 1
            pos = self.steps % 80
            frame = MockState()
            frame.tty_chars = [''.join('@' if col == pos else '.' for col in range(80)) for _ in range(24)]
            frame.tty_colors = [[7]*80 for _ in range(24)]
            frame.glyphs = [[0]*79 for _ in range(21)]

            if self.steps < 50:
                return StepInfo(frame, 'S', 0.0, [], {"Actions": [], "Disallowed": []})
            if self.steps < 55:
                return StepInfo(frame, 'N', 0.1, [('test', 0.1)], {"Actions": [], "Disallowed": []})
            return StepInfo(frame, 'S', 1.0, [('test', 1.0)], {"Actions": [], "Disallowed": []}, ending='DemoEnd')

        def set_model(self, model_path: str) -> None:
            print(f"Model set to: {model_path}")

    run_gui(DemoCtrl(), model_path="models/")
