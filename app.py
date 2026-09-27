"""Surgeon_Tracker — real-time pose estimation and joint-angle overlay for
surgical training video, built on MediaPipe Pose.

Two entry points, both driven by the same tracking loop:
  - "Capture Your Video": live webcam feed
  - "Upload Your Video": a recorded .mp4/.mov file

For each frame, MediaPipe Pose landmarks are extracted and six joint angles
(left/right shoulder, elbow, wrist) are computed and optionally drawn on the
frame, so a supervisor can review a trainee's arm posture and motion pattern
frame by frame.

Run with:
    python app.py
"""

import tkinter as tk
from tkinter.filedialog import askopenfile

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from angles import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

FRAME_SIZE = (700, 400)
LANDMARK_DRAW_SPEC = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
CONNECTION_DRAW_SPEC = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)


def get_joint_points(landmarks, side: str):
    """Pulls the (hip, shoulder, elbow, wrist, index) landmark coordinates
    for one side ('LEFT' or 'RIGHT') into plain [x, y] pairs."""

    def point(name):
        lm = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_{name}").value]
        return [lm.x, lm.y]

    return {
        "hip": point("HIP"),
        "shoulder": point("SHOULDER"),
        "elbow": point("ELBOW"),
        "wrist": point("WRIST"),
        "index": point("INDEX"),
    }


def compute_angles(landmarks):
    """Returns {'L_shoulder': angle, 'L_elbow': ..., 'R_wrist': ...} for both sides."""
    angles = {}
    for side, prefix in (("LEFT", "L"), ("RIGHT", "R")):
        pts = get_joint_points(landmarks, side)
        angles[f"{prefix}_shoulder"] = calculate_angle(pts["hip"], pts["shoulder"], pts["elbow"])
        angles[f"{prefix}_elbow"] = calculate_angle(pts["shoulder"], pts["elbow"], pts["wrist"])
        angles[f"{prefix}_wrist"] = calculate_angle(pts["elbow"], pts["wrist"], pts["index"])
    return angles, {
        "L_shoulder": get_joint_points(landmarks, "LEFT")["shoulder"],
        "R_shoulder": get_joint_points(landmarks, "RIGHT")["shoulder"],
        "L_elbow": get_joint_points(landmarks, "LEFT")["elbow"],
        "R_elbow": get_joint_points(landmarks, "RIGHT")["elbow"],
        "L_wrist": get_joint_points(landmarks, "LEFT")["wrist"],
        "R_wrist": get_joint_points(landmarks, "RIGHT")["wrist"],
    }


def draw_angle_labels(img, angles, positions, frame_size):
    for key, angle in angles.items():
        px = tuple(np.multiply(positions[key], frame_size).astype(int))
        cv2.putText(img, str(angle), px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


class PoseTrackerApp:
    """Tkinter GUI: pick a video source (webcam or file), stream annotated
    frames to a canvas, with a toggle for the joint-angle overlay."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Pose_Estimation_App")
        self.window.geometry("1024x690")
        self.window.config(bg="#00bcb2")

        self.video_playing = False
        self.show_angles = False

        self._build_main_page()

    def run(self):
        self.window.mainloop()

    # -- Main menu -----------------------------------------------------

    def _build_main_page(self):
        self.title = tk.Label(self.window, text="Surgeon Pose Estimation", bg="#00bcb2", font="Times 25")
        self.title.place(relx=0.5, rely=0.1, anchor="center")

        self.capture_btn = tk.Button(self.window, text="Capture Your Video", pady=5, command=self._start_capture)
        self.capture_btn.place(relx=0.31, rely=0.2, anchor="sw")

        self.upload_btn = tk.Button(self.window, text="Upload Your Video", pady=5, command=self._start_upload)
        self.upload_btn.place(relx=0.61, rely=0.2, anchor="sw")

    def _clear_main_page(self):
        self.title.destroy()
        self.capture_btn.destroy()
        self.upload_btn.destroy()

    # -- Playback --------------------------------------------------------

    def _start_capture(self):
        self._clear_main_page()
        self._play_source(cv2.VideoCapture(0))

    def _start_upload(self):
        self._clear_main_page()
        file = askopenfile(mode="r", filetypes=[("Video Files", ["*.mp4", "*.mov"])])
        if file is None:
            self._build_main_page()
            return
        self._play_source(cv2.VideoCapture(str(file.name)))

    def _play_source(self, cap):
        canvas = tk.Canvas(self.window, width=FRAME_SIZE[0], height=FRAME_SIZE[1], bg="#00bcb2")
        canvas.pack()
        self.video_playing = True
        self.show_angles = False

        def back():
            self.video_playing = False
            canvas.destroy()
            back_btn.destroy()
            show_btn.destroy()
            hide_btn.destroy()
            self._build_main_page()

        back_btn = tk.Button(self.window, text="Back to Main Page", pady=5, command=back)
        back_btn.place(relx=0.2, rely=0.8, anchor="sw")
        show_btn = tk.Button(self.window, text="Show Angles", pady=5, command=lambda: setattr(self, "show_angles", True))
        show_btn.place(relx=0.6, rely=0.8, anchor="sw")
        hide_btn = tk.Button(self.window, text="Cover Up Angles", pady=5, command=lambda: setattr(self, "show_angles", False))
        hide_btn.place(relx=0.8, rely=0.8, anchor="sw")

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video_playing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, FRAME_SIZE)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = pose.process(img)
                img.flags.writeable = True

                if results.pose_landmarks:
                    if self.show_angles:
                        try:
                            angles, positions = compute_angles(results.pose_landmarks.landmark)
                            draw_angle_labels(img, angles, positions, (640, 480))
                        except (AttributeError, IndexError):
                            pass  # landmark(s) not visible this frame

                    mp_drawing.draw_landmarks(
                        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        LANDMARK_DRAW_SPEC, CONNECTION_DRAW_SPEC,
                    )

                photo = ImageTk.PhotoImage(image=Image.fromarray(img))
                canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                canvas.image = photo
                self.window.update()

        cap.release()


if __name__ == "__main__":
    PoseTrackerApp().run()
