from __future__ import annotations

import sys
from pathlib import Path

from library.detect_seam_angle_file import (BALL_IMAGE_FOLDER, DEFAULT_SAVE_DIR, BallImage,
                                            SeamAngleDetector, SeamMeasurement,
                                            load_ball_images, save_seam_measurement)


class AutomaticSeamAngleDetector(SeamAngleDetector):
    """Finds the seam on the cropped ball images without any clicking."""

    method = "automatic"

    def detect(self, ball_images: list[BallImage]) -> SeamMeasurement | None:
        """Look for the seam on every ball image, then pick the one to use."""
        seams = []
        for image in ball_images:
            seam = self.find_seam(image)
            if seam is not None:
                seams.append(seam)
        return self.choose_seam(seams)

    def find_seam(self, image: BallImage) -> SeamMeasurement | None:
        """
        Find the seam on one ball image, or return None if it cannot be found.

        Load the crop with image.load(), find two points along the seam in crop
        pixels, and return
        SeamMeasurement(image, image.to_frame(x0, y0), image.to_frame(x1, y1), self.method)
        """
        # TODO
        return None

    def choose_seam(self, seams: list[SeamMeasurement]) -> SeamMeasurement | None:
        """
        Pick which seam to use, or return None if there are none. seams has one
        SeamMeasurement for each image find_seam found the seam on.
        """
        # TODO
        return None


def main(argv: list[str] | None = None) -> None:
    """Run the automatic detector on a folder of ball images from an earlier run."""
    arguments = sys.argv[1:] if argv is None else argv
    folder = Path(arguments[0]) if arguments else Path(DEFAULT_SAVE_DIR) / BALL_IMAGE_FOLDER
    seam = AutomaticSeamAngleDetector().detect(load_ball_images(folder))
    if seam is None:
        print("No seam found.")
        return
    angle = "-" if seam.seam_angle_deg is None else f"{seam.seam_angle_deg:+.2f} deg"
    print(f"Calculated seam angle: {angle} (frame {seam.frame_number})")
    print(f"Seam saved to {save_seam_measurement(seam)}")


if __name__ == "__main__":
    main()
