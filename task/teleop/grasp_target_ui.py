"""Isaac Sim omni.ui panel: pick part → set Design D grasp close target."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

_WINDOW_TITLE = "RoCo Grasp Target (Design D)"


class GraspTargetPanel:
    """Buttons for each asset; invokes ``on_select(part_name)``."""

    def __init__(
        self,
        part_names: Sequence[str],
        on_select: Callable[[str], None],
        *,
        on_full_close: Optional[Callable[[], None]] = None,
        get_status: Optional[Callable[[], str]] = None,
    ) -> None:
        self._part_names = list(part_names)
        self._on_select = on_select
        self._on_full_close = on_full_close
        self._get_status = get_status
        self._window = None
        self._status_label = None
        self._build()

    def _build(self) -> None:
        try:
            import omni.ui as ui
        except Exception as exc:  # pragma: no cover - Isaac-only
            print(f"[grasp_target_ui] omni.ui unavailable: {exc}", flush=True)
            return

        self._window = ui.Window(_WINDOW_TITLE, width=320, height=420)
        with self._window.frame:
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Close target = grasp_width_m → joint rad",
                    height=20,
                )
                self._status_label = ui.Label("current: (none)", height=36, word_wrap=True)
                ui.Spacer(height=4)
                with ui.ScrollingFrame(height=300):
                    with ui.VStack(spacing=2):
                        for name in self._part_names:
                            ui.Button(
                                name,
                                height=28,
                                clicked_fn=self._make_click(name),
                            )
                ui.Spacer(height=4)
                if self._on_full_close is not None:
                    ui.Button(
                        "full close (0 rad)",
                        height=28,
                        clicked_fn=lambda: self._on_full_close and self._on_full_close(),
                    )
                ui.Button("refresh status", height=24, clicked_fn=self.refresh_status)
        self.refresh_status()
        print(f"[grasp_target_ui] opened '{_WINDOW_TITLE}'", flush=True)

    def _make_click(self, name: str):
        def _fn():
            self._on_select(name)
            self.refresh_status()

        return _fn

    def refresh_status(self) -> None:
        if self._status_label is None:
            return
        if self._get_status is not None:
            try:
                self._status_label.text = self._get_status()
                return
            except Exception:
                pass
        self._status_label.text = "current: (unknown)"

    def destroy(self) -> None:
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
