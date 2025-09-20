"""Tkinter-based GUI for automated cell detection."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from . import image_utils
from .model_loader import TensorFlowCellDetector
from .structures import DetectionSummary

SUPPORTED_FILETYPES = [
    ("Microscopy images", "*.png *.jpg *.jpeg *.tif *.tiff"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("TIFF", "*.tif *.tiff"),
]


class CellDetectionApp:
    """High-level GUI application that connects the ML model to user workflows."""

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        detector: Optional[TensorFlowCellDetector] = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.root.title("Cell Detection and Annotation")
        self.root.geometry("1100x720")
        self.root.minsize(980, 640)

        self.detector = detector or TensorFlowCellDetector()

        self.selected_paths: List[str] = []
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._annotated_photo: Optional[ImageTk.PhotoImage] = None
        self._base_image: Optional[Image.Image] = None
        self._annotated_image: Optional[Image.Image] = None
        self._summary: Optional[DetectionSummary] = None
        self._metadata: Optional[dict] = None
        self._detection_thread: Optional[threading.Thread] = None

        self.threshold_var = tk.DoubleVar(value=0.5)
        self.include_metadata_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Select microscope images to begin.")

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout configuration
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.root, padding=12)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.columnconfigure(0, weight=1)

        self.content = ttk.Frame(self.root, padding=12)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()

    def _build_sidebar(self) -> None:
        header = ttk.Label(self.sidebar, text="Image Upload", font=("TkDefaultFont", 12, "bold"))
        header.grid(row=0, column=0, sticky="w")

        self.select_button = ttk.Button(
            self.sidebar, text="Select Image(s)", command=self.select_images
        )
        self.select_button.grid(row=1, column=0, pady=(4, 8), sticky="ew")

        self.file_list = tk.Listbox(self.sidebar, height=5, selectmode=tk.SINGLE)
        self.file_list.grid(row=2, column=0, sticky="ew")

        preview_frame = ttk.LabelFrame(self.sidebar, text="Preview", padding=6)
        preview_frame.grid(row=3, column=0, pady=(12, 8), sticky="ew")
        self.preview_label = ttk.Label(preview_frame, text="No image selected", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        controls_frame = ttk.LabelFrame(self.sidebar, text="Detection Settings", padding=6)
        controls_frame.grid(row=4, column=0, pady=(12, 8), sticky="ew")

        ttk.Label(controls_frame, text="Score threshold").pack(anchor="w")
        threshold_scale = ttk.Scale(
            controls_frame,
            from_=0.1,
            to=0.9,
            orient="horizontal",
            variable=self.threshold_var,
        )
        threshold_scale.pack(fill="x", pady=(2, 4))

        threshold_value = ttk.Label(controls_frame, textvariable=self._formatted_threshold())
        threshold_value.pack(anchor="w")

        self.run_button = ttk.Button(
            controls_frame, text="Run Detection", command=self.run_detection
        )
        self.run_button.pack(fill="x", pady=(8, 0))

        export_frame = ttk.LabelFrame(self.sidebar, text="Export", padding=6)
        export_frame.grid(row=5, column=0, pady=(12, 8), sticky="ew")

        self.include_metadata_check = ttk.Checkbutton(
            export_frame,
            text="Include metadata sidecar",
            variable=self.include_metadata_var,
        )
        self.include_metadata_check.pack(anchor="w")

        self.export_button = ttk.Button(
            export_frame, text="Export Annotated Image", command=self.export_annotated_image
        )
        self.export_button.pack(fill="x", pady=(6, 2))

        self.metadata_button = ttk.Button(
            export_frame, text="Export Metadata", command=self.export_metadata_only
        )
        self.metadata_button.pack(fill="x")

        status_label = ttk.LabelFrame(self.sidebar, text="Status", padding=6)
        status_label.grid(row=6, column=0, pady=(12, 0), sticky="ew")
        ttk.Label(status_label, textvariable=self.status_var, wraplength=220, justify="left").pack(
            fill="x"
        )

    def _formatted_threshold(self) -> tk.StringVar:
        var = tk.StringVar()

        def _update(*_: Iterable[object]) -> None:
            var.set(f"Current: {self.threshold_var.get():.2f}")

        _update()
        self.threshold_var.trace_add("write", _update)
        return var

    def _build_content(self) -> None:
        image_frame = ttk.LabelFrame(self.content, text="Annotated Image", padding=6)
        image_frame.grid(row=0, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(
            image_frame,
            text="Annotated results will appear here after detection.",
            anchor="center",
            justify="center",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        summary_frame = ttk.LabelFrame(self.content, text="Detection Summary", padding=6)
        summary_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        summary_frame.columnconfigure(0, weight=1)

        self.summary_label = ttk.Label(summary_frame, justify="left")
        self.summary_label.grid(row=0, column=0, sticky="w")

        table_frame = ttk.Frame(self.content)
        table_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("ID", "Score", "Xmin", "Ymin", "Xmax", "Ymax")
        self.detection_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
        )
        for column in columns:
            self.detection_tree.heading(column, text=column)
            self.detection_tree.column(column, anchor="center", width=90)
        self.detection_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.detection_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.detection_tree.configure(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def select_images(self) -> None:
        file_paths = filedialog.askopenfilenames(
            parent=self.root, title="Select cell image slices", filetypes=SUPPORTED_FILETYPES
        )
        if not file_paths:
            return

        self.selected_paths = list(file_paths)[:3]
        self._update_selected_list()
        self._update_preview()
        self.status_var.set(
            f"Loaded {len(self.selected_paths)} image(s). Run detection when ready."
        )

    def _update_selected_list(self) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.selected_paths:
            self.file_list.insert(tk.END, Path(path).name)

    def _update_preview(self) -> None:
        if not self.selected_paths:
            self.preview_label.configure(text="No image selected", image="")
            self._preview_photo = None
            return

        preview_path = self.selected_paths[1] if len(self.selected_paths) > 1 else self.selected_paths[0]
        try:
            with Image.open(preview_path) as image:
                preview = image.convert("RGB")
            preview.thumbnail((260, 260))
            self._preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as exc:  # pragma: no cover - user feedback path
            self.preview_label.configure(text=f"Preview unavailable:\n{exc}", image="")
            self._preview_photo = None

    def run_detection(self) -> None:
        if not self.selected_paths:
            messagebox.showwarning("No images", "Please select at least one image to analyze.")
            return
        if self._detection_thread and self._detection_thread.is_alive():  # pragma: no cover - UI guard
            messagebox.showinfo("Detection running", "Please wait for the current detection to finish.")
            return

        threshold = float(self.threshold_var.get())
        self.status_var.set("Running detection...")
        self.run_button.configure(state=tk.DISABLED)
        self.export_button.configure(state=tk.DISABLED)
        self.metadata_button.configure(state=tk.DISABLED)

        def task() -> None:
            try:
                stack, base_image, used_paths = image_utils.create_image_stack(self.selected_paths)
                self._base_image = base_image
                predictions = self.detector.predict(stack)
                summary = image_utils.prepare_detection_summary(
                    image_size=stack.shape[:2],
                    detection_boxes=predictions["boxes"],
                    detection_scores=predictions["scores"],
                    threshold=threshold,
                    source_paths=used_paths,
                    model_info=self.detector.model_info,
                )
                annotated = image_utils.draw_detections_on_image(base_image, summary)
                metadata = image_utils.build_metadata(summary)
            except Exception as exc:  # pragma: no cover - integration path
                traceback.print_exc()
                self.root.after(0, lambda: self._on_detection_error(exc))
                return

            self.root.after(
                0,
                lambda: self._on_detection_complete(
                    annotated_image=annotated, summary=summary, metadata=metadata
                ),
            )

        self._detection_thread = threading.Thread(target=task, daemon=True)
        self._detection_thread.start()

    def _on_detection_complete(
        self, annotated_image: Image.Image, summary: DetectionSummary, metadata: dict
    ) -> None:
        self._annotated_image = annotated_image
        self._summary = summary
        self._metadata = metadata

        self._update_result_display()
        self._populate_detection_table(summary)

        self.run_button.configure(state=tk.NORMAL)
        self.export_button.configure(state=tk.NORMAL)
        self.metadata_button.configure(state=tk.NORMAL)

        self.status_var.set(f"Detection complete. Identified {summary.cell_count} cell(s).")

    def _on_detection_error(self, error: Exception) -> None:
        self.run_button.configure(state=tk.NORMAL)
        self.export_button.configure(state=tk.NORMAL)
        self.metadata_button.configure(state=tk.NORMAL)
        messagebox.showerror("Detection failed", str(error))
        self.status_var.set("Detection failed. See console for details.")

    def _update_result_display(self) -> None:
        if not self._annotated_image or not self._summary:
            return

        display = self._annotated_image.copy()
        display.thumbnail((780, 780))
        self._annotated_photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self._annotated_photo, text="")

        scores = [det.score for det in self._summary.detections]
        avg_score = mean(scores) if scores else 0.0
        width, height = self._summary.image_size
        summary_text = (
            f"Cells detected: {self._summary.cell_count}\n"
            f"Average confidence: {avg_score:.2f}\n"
            f"Image size: {width} x {height} pixels\n"
            f"Threshold: {self._summary.detection_threshold:.2f}"
        )
        self.summary_label.configure(text=summary_text)

    def _populate_detection_table(self, summary: DetectionSummary) -> None:
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
        for detection in summary.detections:
            x0, y0, x1, y1 = detection.bbox
            self.detection_tree.insert(
                "",
                "end",
                values=(
                    detection.id,
                    f"{detection.score:.2f}",
                    x0,
                    y0,
                    x1,
                    y1,
                ),
            )

    def export_annotated_image(self) -> None:
        if not self._annotated_image or not self._summary:
            messagebox.showinfo("No results", "Run detection before exporting.")
            return

        source_path = Path(self._summary.source_paths[0])
        initial_name = f"{source_path.stem}_annotated.png"
        output_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export annotated image",
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("TIFF", "*.tif;*.tiff")],
        )
        if not output_path:
            return

        try:
            self._annotated_image.save(output_path)
            if self.include_metadata_var.get() and self._metadata:
                metadata_path = Path(output_path).with_name(
                    f"{Path(output_path).stem}_metadata.json"
                )
                image_utils.save_metadata(self._metadata, metadata_path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            messagebox.showerror("Export failed", str(exc))
            return

        messagebox.showinfo("Export complete", "Annotated image exported successfully.")

    def export_metadata_only(self) -> None:
        if not self._metadata:
            messagebox.showinfo("No metadata", "Run detection before exporting metadata.")
            return

        output_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export metadata",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="cell_detections.json",
        )
        if not output_path:
            return

        try:
            image_utils.save_metadata(self._metadata, Path(output_path))
        except Exception as exc:  # pragma: no cover - filesystem errors
            messagebox.showerror("Export failed", str(exc))
            return

        messagebox.showinfo("Export complete", "Metadata exported successfully.")

    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.root.mainloop()


def main() -> None:  # pragma: no cover - entry point
    """Entry point used by ``python -m cell_detection_gui``."""

    app = CellDetectionApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
