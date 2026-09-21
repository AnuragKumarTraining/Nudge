from collections import defaultdict
from config import AppConfig

class StateTracker:
    """Tracks state durations, transitions, and visit occurrences."""
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.duration: defaultdict[str, float] = defaultdict(float)
        self.visits: defaultdict[str, int] = defaultdict(int)
        self.committed: str | None = None
        self.candidate: str | None = None
        self.candidate_since: int | None = None
        self.blink_since: int | None = None
        self.last_ts: int | None = None

    def update(self, raw_state: str, ts_ms: int) -> None:
        if self.last_ts is not None and self.committed not in (None, *self.cfg.ignored_states):
            delta_sec = max(0.0, (ts_ms - self.last_ts) / 1000.0)
            self.duration[self.committed] += delta_sec
        self.last_ts = ts_ms

        if raw_state == self.cfg.blink:
            if self.blink_since is None:
                self.blink_since = ts_ms
            if (
                self.committed is not None
                and (ts_ms - self.blink_since) / 1000.0 < self.cfg.blink_hold_sec
            ):
                self.candidate = None
                return
        else:
            self.blink_since = None

        if raw_state == self.committed:
            self.candidate = None
            return

        if raw_state != self.candidate:
            self.candidate = raw_state
            self.candidate_since = ts_ms
            return

        if (ts_ms - self.candidate_since) / 1000.0 >= self.cfg.min_dwell_sec:
            self.committed = raw_state
            self.candidate = None
            if raw_state not in self.cfg.ignored_states:
                self.visits[raw_state] += 1

    def export_report(self, video_source: str) -> dict:
        metrics = [
            {
                "gaze_direction": direction,
                "total_time_seconds": round(self.duration[direction], 2),
                "occurrences": self.visits[direction],
            }
            for direction in sorted(self.visits)
            if direction not in self.cfg.ignored_states
        ]
        return {
            "video_source": video_source,
            "total_tracked_seconds": round(sum(m["total_time_seconds"] for m in metrics), 2),
            "metrics": metrics,
        }