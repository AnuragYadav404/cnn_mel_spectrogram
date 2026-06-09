# Base CNN Spectrogram

Hydra-based refactor of the ESC-50 spectrogram classifier.

## Layout

- `main.py`: Hydra entrypoint
- `src/esc50/data`: metadata, splits, dataset
- `src/esc50/models`: pooling and CNN classifier
- `src/esc50/training`: Lightning wrapper and experiment runner
- `src/esc50/inference`: checkpoint loading and prediction helpers
- `configs/`: Hydra config tree

## Run

```bash
python main.py
```

Or from the script wrapper:

```bash
python scripts/train.py
```
