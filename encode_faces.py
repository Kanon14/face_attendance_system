import argparse
import pickle
from pathlib import Path

from deepface import DeepFace


REGISTERED_FACES_DIR = Path("data") / "registered_faces"
ENCODINGS_DIR = Path("data") / "encodings"
ENCODINGS_FILE = ENCODINGS_DIR / "face_encodings.pkl"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_registered_images() -> list[tuple[str, Path]]:
    registered_images = []

    if not REGISTERED_FACES_DIR.exists():
        return registered_images

    for worker_dir in sorted(REGISTERED_FACES_DIR.iterdir()):
        if not worker_dir.is_dir():
            continue

        for image_path in sorted(worker_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                registered_images.append((worker_dir.name, image_path))

    return registered_images


def encode_faces(model_name: str, detector_backend: str) -> None:
    registered_images = get_registered_images()

    if not registered_images:
        print(f"Error: no images found in {REGISTERED_FACES_DIR}.")
        return

    names = []
    encodings = []
    skipped_count = 0

    for worker_name, image_path in registered_images:
        try:
            results = DeepFace.represent(
                img_path=str(image_path),
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=True,
            )
        except ValueError:
            skipped_count += 1
            print(f"Skipped {image_path}: no face detected.")
            continue

        if not results:
            skipped_count += 1
            print(f"Skipped {image_path}: no face detected.")
            continue

        if len(results) > 1:
            print(
                f"Warning: multiple faces found in {image_path}. "
                "Using the first detected face."
            )

        names.append(worker_name)
        encodings.append(results[0]["embedding"])
        print(f"Encoded {image_path}")

    if not encodings:
        print("Error: no face encodings were generated.")
        return

    ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)
    with ENCODINGS_FILE.open("wb") as file:
        pickle.dump({"names": names, "encodings": encodings}, file)

    print(f"Saved {len(encodings)} face encoding(s) to {ENCODINGS_FILE}")
    if skipped_count:
        print(f"Skipped {skipped_count} image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate face encodings from registered worker images."
    )
    parser.add_argument(
        "--model",
        default="Facenet",
        help="DeepFace model name. Example: Facenet, VGG-Face, ArcFace.",
    )
    parser.add_argument(
        "--detector",
        default="opencv",
        help="DeepFace detector backend. Example: opencv, retinaface, mtcnn.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_faces(args.model, args.detector)
