import os
import cv2
import argparse
import numpy as np
from pathlib import Path

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_image(image, width=700):
    h, w = image.shape[:2]
    if w <= width:
        return image
    scale = width / float(w)
    return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def morphological_skeleton(binary_img):
    img = (binary_img > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def thin_image(binary_img):
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(binary_img)
    return morphological_skeleton(binary_img)


def segment_foreground(enhanced):
    # Fingerprint photos from phone often benefit from adaptive thresholding.
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def preprocess_fingerprint(image):
    steps = {}
    original = resize_image(image, width=700)
    steps['01_original'] = original

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    steps['02_grayscale'] = gray

    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['03_denoised'] = denoised

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    steps['04_enhanced'] = enhanced

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    steps['05_sharpened'] = sharpened

    segmented = segment_foreground(sharpened)
    steps['06_segmented'] = segmented

    sobel_x = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    steps['07_sobel'] = sobel

    canny = cv2.Canny(sharpened, 50, 150)
    steps['08_canny'] = canny

    skeleton = thin_image(segmented)
    steps['09_skeleton'] = skeleton

    minutiae_vis, endings, bifurcations = detect_minutiae(skeleton)
    steps['10_minutiae'] = minutiae_vis

    return steps, endings, bifurcations


def detect_minutiae(skeleton):
    skel = (skeleton > 0).astype(np.uint8)
    vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    endings = []
    bifurcations = []
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 1:
                neighborhood = skel[y - 1:y + 2, x - 1:x + 2]
                count = int(np.sum(neighborhood)) - 1
                if count == 1:
                    endings.append((x, y))
                elif count >= 3:
                    bifurcations.append((x, y))

    # Suppress dense duplicate points by simple distance filtering.
    endings = filter_close_points(endings, min_distance=8)
    bifurcations = filter_close_points(bifurcations, min_distance=8)

    for x, y in endings:
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    for x, y in bifurcations:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    cv2.putText(vis, f'Endings: {len(endings)}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f'Bifurcations: {len(bifurcations)}', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis, endings, bifurcations


def filter_close_points(points, min_distance=8):
    kept = []
    for x, y in points:
        ok = True
        for kx, ky in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 < min_distance ** 2:
                ok = False
                break
        if ok:
            kept.append((x, y))
    return kept


def to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def make_step_grid(steps, output_path: Path):
    labels = list(steps.keys())
    images = [to_bgr(steps[k]) for k in labels]
    tile_w, tile_h = 360, 260
    rendered = []
    for label, image in zip(labels, images):
        canvas = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        h, w = image.shape[:2]
        scale = min((tile_w - 20) / w, (tile_h - 50) / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0 = 40 + (tile_h - 40 - new_h) // 2
        x0 = (tile_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        cv2.putText(canvas, label.replace('_', ' '), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        rendered.append(canvas)

    rows = []
    for i in range(0, len(rendered), 3):
        row_imgs = rendered[i:i + 3]
        while len(row_imgs) < 3:
            row_imgs.append(np.full((tile_h, tile_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)
    cv2.imwrite(str(output_path), grid)


def save_steps(base_name: str, steps: dict, endings, bifurcations, output_root: Path):
    sample_dir = output_root / base_name
    ensure_dir(sample_dir)
    for step_name, image in steps.items():
        out_path = sample_dir / f'{step_name}.png'
        cv2.imwrite(str(out_path), image)

    make_step_grid(steps, sample_dir / f'{base_name}_overview.png')

    with open(sample_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'Image: {base_name}\n')
        f.write(f'Ridge endings: {len(endings)}\n')
        f.write(f'Bifurcations: {len(bifurcations)}\n')
        f.write('Green points = ridge endings\n')
        f.write('Red points = bifurcations\n')


def process_folder(input_dir: Path, output_dir: Path):
    ensure_dir(output_dir)
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not files:
        raise FileNotFoundError(
            f'No image files found in {input_dir}. Put your fingerprint photos into the pic folder.'
        )

    all_results = []
    for file_path in files:
        image = cv2.imread(str(file_path))
        if image is None:
            print(f'Skipped unreadable file: {file_path.name}')
            continue
        steps, endings, bifurcations = preprocess_fingerprint(image)
        base_name = file_path.stem
        save_steps(base_name, steps, endings, bifurcations, output_dir)
        all_results.append((file_path.name, len(endings), len(bifurcations)))
        print(f'Processed {file_path.name}: endings={len(endings)}, bifurcations={len(bifurcations)}')

    report_path = output_dir / 'report.csv'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('file_name,ridge_endings,bifurcations\n')
        for row in all_results:
            f.write(f'{row[0]},{row[1]},{row[2]}\n')
    print(f'\nDone. Results saved in: {output_dir}')
    print(f'Summary report: {report_path}')


def main():
    parser = argparse.ArgumentParser(description='Fingerprint enhancement and minutiae extraction pipeline.')
    parser.add_argument('--input', default='pic', help='Folder containing fingerprint images.')
    parser.add_argument('--output', default='output', help='Folder to save processed results.')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    process_folder(input_dir, output_dir)


if __name__ == '__main__':
    main()
