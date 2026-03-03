import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class JigsawEdgeDetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Jigsaw Edge Piece Detector")
        self.root.geometry("980x720")

        self.image_bgr: np.ndarray | None = None
        self.display_image = None

        toolbar = tk.Frame(root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        upload_btn = tk.Button(toolbar, text="Upload Image", command=self.load_image)
        upload_btn.pack(side=tk.LEFT, padx=8, pady=8)

        detect_btn = tk.Button(toolbar, text="Detect Edge Pieces", command=self.detect_edges)
        detect_btn.pack(side=tk.LEFT, padx=8, pady=8)

        self.status = tk.Label(toolbar, text="Load an image to begin.")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(root, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Load error", "Could not open the selected image.")
            return
        self.image_bgr = image
        self.status.config(text="Image loaded. Click Detect Edge Pieces.")
        self.show_image(image)

    def detect_edges(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("No image", "Please upload an image first.")
            return

        processed, count = self.process_image(self.image_bgr)
        self.status.config(text=f"Detected {count} edge pieces.")
        self.show_image(processed)

    def process_image(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
        )

        contours = self.segment_pieces(thresh)

        result = image_bgr.copy()
        edge_piece_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:  # skip small noise
                continue

            if self.is_edge_piece(contour):
                color = (0, 220, 0)
                edge_piece_count += 1
            else:
                color = (0, 0, 220)

            cv2.drawContours(result, [contour], -1, color, 3)

        return result, edge_piece_count

    def segment_pieces(self, binary: np.ndarray) -> list[np.ndarray]:
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opened, kernel, iterations=3)
        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(bgr, markers)

        contours = []
        for label in range(2, markers.max() + 1):
            mask = np.zeros(binary.shape, dtype=np.uint8)
            mask[markers == label] = 255
            piece_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(piece_contours)

        return contours

    def is_edge_piece(self, contour: np.ndarray) -> bool:
        rect = cv2.boundingRect(contour)
        max_span = max(rect[2], rect[3])
        if max_span == 0:
            return False
        x, y, w, h = rect
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        shifted = contour - [x, y]
        cv2.drawContours(mask, [shifted], -1, 255, 1)

        edges = cv2.Canny(mask, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=10)

        longest = 0.0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length > longest:
                    longest = length

        # Count concave defects; edge pieces should have fewer deep indentations.
        defects_count = 0
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) >= 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                depth_threshold = 0.02 * max_span * 256
                for defect in defects:
                    depth = defect[0][3]
                    if depth >= depth_threshold:
                        defects_count += 1

        long_line_threshold = 0.3 * max_span
        return longest >= long_line_threshold and defects_count <= 4

    def show_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 900, 600

        image.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        self.display_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.display_image
        )


def main() -> None:
    root = tk.Tk()
    app = JigsawEdgeDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
