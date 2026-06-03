import argparse
import re
from pathlib import Path

import cv2


REGISTERED_FACES_DIR = Path("data") / "registered_faces"


def clean_worker_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    cleaned = cleaned.strip("_")

    if not cleaned:
        raise ValueError("Worker name cannot be empty.")

    return cleaned


def get_next_image_number(worker_dir: Path, worker_name: str) -> int:
    existing_numbers = []

    for image_path in worker_dir.glob(f"{worker_name}_*.jpg"):
        number_text = image_path.stem.removeprefix(f"{worker_name}_")
        if number_text.isdigit():
            existing_numbers.append(int(number_text))

    if not existing_numbers:
        return 1

    return max(existing_numbers) + 1


def register_faces(camera_index: int, width: int, height: int) -> None:
    worker_name = clean_worker_name(input("Enter worker name: "))
    worker_dir = REGISTERED_FACES_DIR / worker_name
    worker_dir.mkdir(parents=True, exist_ok=True)

    next_image_number = get_next_image_number(worker_dir, worker_name)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    print("Webcam opened.")
    print("Press C to capture an image.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read from webcam.")
                break

            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Worker: {worker_name} | Captured: {next_image_number - 1}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press C to capture | Press Q to quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Register Faces", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("c"):
                image_name = f"{worker_name}_{next_image_number:03d}.jpg"
                image_path = worker_dir / image_name
                cv2.imwrite(str(image_path), frame)
                print(f"Saved {image_path}")
                next_image_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register worker face images using a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Webcam frame width.")
    parser.add_argument("--height", type=int, default=720, help="Webcam frame height.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_faces(args.camera, args.width, args.height)
