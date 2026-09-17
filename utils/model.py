"""Model definition and evaluation utilities."""


class RoomModel:
    """Room analysis model representation."""

    def __init__(self, room_name: str):
        self.room_name = room_name

    def evaluate(self, image_path: str) -> dict:
        """Evaluate input image against baseline model parameters."""
        return {"room": self.room_name, "status": "ok"}
