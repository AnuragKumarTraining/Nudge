import json
import time
import cv2
import mediapipe as mp

from config import AppConfig
from gaze import Calibrator, GazeClassifier, extract_face_signals
from model_loader import ensure_model_exists
from tracker import StateTracker
from visualizer import Visualizer

def main():
    config = AppConfig()
    ensure_model_exists(config.model_path, config.model_url)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=config.model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot initialize capture from source: {config.video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if (fps and 1 < fps <= 240) else 30.0

    calibrator = Calibrator(config.calibration_frames)
    classifier = GazeClassifier(config)
    tracker = StateTracker(config)
    visualizer = Visualizer(config)

    if config.is_live:
        calibrator.start()

    t0 = time.monotonic()
    frame_idx, last_ts = 0, -1
    dbg_data = {"h": 0.0, "v": 0.0, "yaw": 0.0, "pitch": 0.0}

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h_px, w_px = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.monotonic() - t0) * 1000) if config.is_live else int(frame_idx * 1000.0 / fps)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            frame_idx += 1

            result = landmarker.detect_for_video(mp_img, ts)
            raw_state = config.no_face

            has_face = bool(
                result.face_landmarks
                and result.face_blendshapes
                and result.facial_transformation_matrixes
            )

            if has_face:
                visualizer.draw_landmarks(frame, result.face_landmarks[0], w_px, h_px)
                h, v, blink, yaw, pitch = extract_face_signals(result, config)

                dbg_data = {
                    "h": h - calibrator.baseline["h"],
                    "v": v - calibrator.baseline["v"],
                    "yaw": yaw - calibrator.baseline["yaw"],
                    "pitch": pitch - calibrator.baseline["pitch"],
                }

                if calibrator.active:
                    raw_state = config.calibrating
                    if blink <= config.blink_threshold:
                        calibrator.update(h, v, yaw, pitch)
                    classifier.reset()
                elif blink > config.blink_threshold:
                    raw_state = config.blink
                else:
                    raw_state = classifier.classify(h, v, yaw, pitch, calibrator.baseline)
            else:
                classifier.reset()

            tracker.update(raw_state, ts)
            visualizer.draw_hud(frame, tracker, calibrator, dbg_data, h_px)

            cv2.imshow("Iris Gaze Analytics", frame)
            delay = 1 if config.is_live else max(1, int(1000.0 / fps))
            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                calibrator.start()
                classifier.reset()

    cap.release()
    cv2.destroyAllWindows()

    report = tracker.export_report(str(config.video_source))
    print(json.dumps(report, indent=4))
    with open(config.json_output_path, "w") as out:
        json.dump(report, out, indent=4)

if __name__ == "__main__":
    main()