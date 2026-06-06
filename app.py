import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path

loaded_model = None
model_path_str = ""

# =========================
# LOAD YOLO MODEL
# =========================

def load_model(model_file):

    global loaded_model, model_path_str

    if model_file is None:
        return (
            "❌ Please select a best.pt model file.",
            gr.update(interactive=False)
        )

    try:

        from ultralytics import YOLO

        loaded_model = YOLO(model_file.name)

        model_path_str = model_file.name

        name = Path(model_file.name).name

        return (
            f"✅ Model Loaded Successfully: {name}",
            gr.update(interactive=True)
        )

    except Exception as e:

        loaded_model = None

        return (
            f"❌ Error Loading Model: {str(e)}",
            gr.update(interactive=False)
        )


def draw_boxes(frame, results, conf_threshold):

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            if conf < conf_threshold:
                continue

            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Class info
            cls_id = int(box.cls[0])
            class_name = loaded_model.names[cls_id]

            if conf >= 0.65:

                color = (0, 255, 0)

                label = f"{class_name} {conf:.0%}"

            else:

                color = (0, 0, 255)

                label = f"Unknown {conf:.0%}"

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Label size
            (lw, lh), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - lh - 10),
                (x1 + lw + 10, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            detections.append(conf)

    return frame, detections


def process_image(image, conf_threshold):

    if loaded_model is None:
        return None, "⚠️ Please load the model first."

    if image is None:
        return None, "⚠️ Please upload an image."

    try:

        frame = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        start_time = time.time()

        results = loaded_model(
            frame,
            conf=conf_threshold,
            verbose=False
        )

        inference_time = (time.time() - start_time) * 1000

        frame, detections = draw_boxes(
            frame,
            results,
            conf_threshold
        )

        # Top overlay
        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 50),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.5,
            frame,
            0.5,
            0,
            frame
        )

        # Stats text
        stats_text = (
            f"Faces: {len(detections)} | "
            f"Inference: {inference_time:.0f} ms"
        )

        cv2.putText(
            frame,
            stats_text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        out_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Summary
        if detections:

            avg_conf = np.mean(detections)
            max_conf = np.max(detections)

            summary = f"""


- Faces Detected: {len(detections)}
- Average Confidence: {avg_conf:.1%}
- Highest Confidence: {max_conf:.1%}
- Inference Time: {inference_time:.0f} ms
"""

        else:

            summary = """


Try lowering the confidence threshold.
"""

        return out_rgb, summary

    except Exception as e:

        return None, f" Error: {str(e)}"


def process_video(video_file, conf_threshold, progress=gr.Progress()):

    if loaded_model is None:
        return None, "⚠️ Please load the model first."

    if video_file is None:
        return None, "⚠️ Please upload a video."

    try:

        cap = cv2.VideoCapture(video_file)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output path
        out_path = tempfile.mktemp(suffix=".mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            fps,
            (width, height)
        )

        all_confidences = []

        frame_number = 0

        total_detections = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = loaded_model(
                frame,
                conf=conf_threshold,
                verbose=False
            )

            frame, detections = draw_boxes(
                frame,
                results,
                conf_threshold
            )

            all_confidences.extend(detections)

            total_detections += len(detections)

            # Frame counter
            frame_text = f"Frame {frame_number+1}/{total_frames}"

            cv2.putText(
                frame,
                frame_text,
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            frame_number += 1

            if total_frames > 0:

                progress(
                    frame_number / total_frames,
                    desc=f"Processing frame {frame_number}/{total_frames}"
                )

        cap.release()

        writer.release()

        # Video summary
        if all_confidences:

            avg_conf = np.mean(all_confidences)

            summary = f"""

- Total Frames Processed: {frame_number}
- Total Face Detections: {total_detections}
- Average Confidence: {avg_conf:.1%}
- Video Duration: {frame_number/fps:.1f} seconds
"""

        else:

            summary = """
"""

        return out_path, summary

    except Exception as e:

        return None, f"❌ Error: {str(e)}"

css = """

body, .gradio-container {
    background: #0b0f19 !important;
    color: white !important;
    font-family: Arial !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

button {
    border-radius: 8px !important;
}

"""


with gr.Blocks(
    css=css,
    title="Face Detection Security System"
) as demo:

    # Header
    gr.Markdown(
        """
# Face Detection & Recognition Security System

YOLOv8 Transfer Learning Based Security Application
"""
    )

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=1):

            model_file = gr.File(
                label="Upload best.pt Model",
                file_types=[".pt"]
            )

            load_btn = gr.Button(
                "LOAD MODEL",
                variant="primary"
            )

            model_status = gr.Markdown(
                "_Model not loaded_"
            )

            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.40,
                step=0.05,
                label="Confidence Threshold",
                interactive=False
            )

        # RIGHT PANEL
        with gr.Column(scale=3):

            with gr.Tabs():

                # IMAGE TAB
                with gr.TabItem("IMAGE DETECTION"):

                    with gr.Row():

                        img_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            interactive=False,
                            height=350
                        )

                        img_output = gr.Image(
                            label="Detection Result",
                            type="numpy",
                            height=350
                        )

                    img_btn = gr.Button(
                        "DETECT FACES",
                        variant="primary",
                        interactive=False
                    )

                    img_stats = gr.Markdown(
                        "_Detection results will appear here_"
                    )

                # VIDEO TAB
                with gr.TabItem("VIDEO DETECTION"):

                    with gr.Row():

                        vid_input = gr.Video(
                            label="Upload Video",
                            interactive=False,
                            height=320
                        )

                        vid_output = gr.Video(
                            label="Detection Result",
                            height=320
                        )

                    vid_btn = gr.Button(
                        "DETECT IN VIDEO",
                        variant="primary",
                        interactive=False
                    )

                    vid_stats = gr.Markdown(
                        "_Video detection results will appear here_"
                    )


    def on_load(f):

        status, slider_state = load_model(f)

        return (
            status,
            slider_state,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

    load_btn.click(
        on_load,
        inputs=[model_file],
        outputs=[
            model_status,
            conf_slider,
            img_input,
            img_btn,
            vid_input,
            vid_btn
        ]
    )

    img_btn.click(
        process_image,
        inputs=[img_input, conf_slider],
        outputs=[img_output, img_stats]
    )

    vid_btn.click(
        process_video,
        inputs=[vid_input, conf_slider],
        outputs=[vid_output, vid_stats]
    )


if __name__ == "__main__":

    print("=" * 60)
    print(" FACE DETECTION SECURITY SYSTEM ")
    print("=" * 60)

    print("\nStarting server...")

    print("\nOpen in browser:")
    print("http://127.0.0.1:7860")

    print("\nPress CTRL+C to stop.\n")

    demo.launch(inbrowser=True)