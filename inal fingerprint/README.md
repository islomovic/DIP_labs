# Final Fingerprint

This project reads fingerprint photos from the `pic` folder and processes them step by step:

1. Original image
2. Grayscale conversion
3. Noise removal
4. Contrast enhancement (CLAHE)
5. Sharpening / ridge clarification
6. Segmentation / binary thresholding
7. Sobel edge detection
8. Canny edge detection
9. Skeleton / thinning
10. Minutiae candidate detection

## Folder structure

- `pic/` -> put your fingerprint photos here
- `output/` -> results will be saved here after running the code
- `fingerprint_pipeline.py` -> main Python program
- `requirements.txt` -> Python packages

## How to use on Google Colab

### Option A: Upload this zip to Colab and unzip it

```python
!unzip "final fingerprint.zip"
%cd "final fingerprint"
!pip install -r requirements.txt
```

Upload your photos into the `pic` folder, then run:

```python
!python fingerprint_pipeline.py --input pic --output output
```

To view saved overview images:

```python
import os
from IPython.display import Image, display

for name in sorted(os.listdir('output')):
    overview = os.path.join('output', name, f'{name}_overview.png')
    if os.path.exists(overview):
        display(Image(filename=overview))
```

## How to use on your computer

1. Install Python 3.
2. Open terminal in this project folder.
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Put your fingerprint photos inside the `pic` folder.
5. Run:

```bash
python fingerprint_pipeline.py --input pic --output output
```

## Notes

- Better photos give better results.
- Use macro mode, good light, and sharp focus.
- Green points are ridge endings.
- Red points are bifurcations.
- Raw Mac Touch ID fingerprint images are not available to apps, so this project works with photo images that you capture yourself.
