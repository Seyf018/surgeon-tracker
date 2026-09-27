# Surgeon_Tracker

A desktop app for reviewing surgical-training video: tracks a trainee's
upper-body pose in real time and overlays shoulder/elbow/wrist joint angles,
so a supervisor can review posture and motion frame by frame — from a live
webcam feed or an uploaded recording.

Built during a computer-vision internship at LMS-ENISo, reviewing pose
tracking tools for surgeon training supervision.

## How it works

- **Pose backend:** [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
  (chosen over OpenPose after comparing both — MediaPipe ran comfortably in
  real time on CPU, which OpenPose did not).
- For each frame: extract pose landmarks → compute 6 joint angles (left/right
  shoulder, elbow, wrist, via the standard three-point angle formula) →
  optionally draw them on the frame → render to a Tkinter canvas.
- Works on a live webcam feed or an uploaded `.mp4`/`.mov` file, with an
  on/off toggle for the angle overlay so the raw tracked skeleton can be
  reviewed on its own.

## Repo structure

```
app.py       # Tkinter GUI + MediaPipe tracking loop
angles.py    # three-point joint-angle geometry
```

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Pick "Capture Your Video" for a live webcam feed, or "Upload Your Video" for
a recorded file. "Show Angles" / "Cover Up Angles" toggles the overlay.

## Notes on this cleanup

This is a reorganized version of the original 2023 internship code. Several
earlier prototype scripts (separate webcam-preview and single-source
tracking tests) led up to a final Tkinter app with both a live-capture and
an upload path — those two paths were implemented as two near-identical
~90-line blocks of copy-pasted code. This version keeps the same tracking
logic and UI flow but factors the shared per-frame processing (landmark
extraction, angle computation, drawing, canvas update) into one path used by
both capture modes, and pulls the angle math into its own module.

The tracking model and math are unchanged from the original — this is a
reorganization pass, not a retrain or a redesign.

## License

MIT
