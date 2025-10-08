# Noise2Noise for Stress–Strain Curves

This how-to shows how to denoise repeated stress–strain (force–displacement) measurements of the same specimen using the Noise2Noise workflow implemented in Auto-Denoise.

The workflow mirrors the spectral tutorial but adapts the data-loading and plotting steps to 1D mechanical test signals. It assumes you recorded multiple noisy realizations (e.g., eight replicates) of the same loading path.

## 1. Organize the Replicate Measurements

Collect the measurements into a single NumPy array with shape `(n_realizations, n_points)`. Each row is one acquisition; the columns are the synchronized strain (or displacement) samples.

```python
from pathlib import Path

import numpy as np
import pandas as pd

# Suppose each CSV has columns: displacement_mm, force_kN
# Load the eight replicates and align them on the same displacement grid
replicate_paths = sorted(Path("data/stress_strain/").glob("sample_repeat_*.csv"))
curves = [pd.read_csv(path) for path in replicate_paths]

# If the displacement grids differ slightly, interpolate onto a common axis
common_disp = np.linspace(0.0, 12.0, 2048)  # mm
forces = np.stack([
    np.interp(common_disp, curve["displacement_mm"], curve["force_kN"])
    for curve in curves
])
```

!!! tip
    You can skip the interpolation when the samples are already aligned and share the same number of points.

The resulting `forces` array is what you will feed into the denoiser; its shape will be `(8, 2048)` when you have eight repeats resampled to 2048 points each.

## 2. Instantiate a 1D Denoising Network

Noise2Noise works with any of the 1D architectures exposed by Auto-Denoise. A compact DnCNN is often sufficient for stress–strain signals.

```python
import autoden as ad

model = ad.NetworkParamsDnCNN(n_dims=1, n_features=8, n_layers=8).get_model()
```

Feel free to tune the number of features and layers to trade off between fidelity and training time. Small models (4–8 features, 6–10 layers) usually capture the smooth stress–strain trend without overfitting.

## 3. Prepare Input–Target Pairs for Noise2Noise

Use `N2N.prepare_data` to build self-supervised pairs. With eight realizations, the method automatically forms leave-one-out targets where every curve is predicted from the mean of the remaining seven.

```python
n2n = ad.N2N(model=model)
inp, tgt, mask = n2n.prepare_data(forces, num_tst_ratio=0.1)
```

Key parameters:

- `forces` can have any leading number of realizations ≥2; you already satisfy this with eight repeats.
- `num_tst_ratio` controls how many samples are held out to monitor the loss. Set it near `0.1` when you want to reserve roughly 10% of the points.
- Switch `strategy="X:1"` when you prefer the current curve to act as the target and the averaged peers to be the input; the default `"1:X"` works well for most cases.【F:src/autoden/algorithms/noise2noise.py†L18-L76】

## 4. Train the Denoiser

Train until the validation loss plateaus. Because the signal is 1D, you can often use a relatively aggressive learning rate.

```python
losses = n2n.train(
    inp,
    tgt,
    mask,
    epochs=2_000,
    learning_rate=5e-3,
    lower_limit=0.0,  # clamp predictions if the force is non-negative
    restarts=1,
)
```

Monitoring the logged losses lets you stop early once the curve stabilizes.【F:docs/tutorials/denoising_spectra.md†L38-L65】

## 5. Denoise Any Replicate or Their Average

During inference, supply the same stacked array of noisy curves. By default, `infer` averages the individual predictions along the realization axis; set `average_splits=False` if you want separate denoised outputs per replicate.

```python
predicted_force = n2n.infer(inp)
```

You can then plot the clean curve against the original noisy measurements.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(common_disp, forces[0], label="Replicate 1", alpha=0.5)
ax.plot(common_disp, predicted_force, label="Noise2Noise prediction", linewidth=2)
ax.set_xlabel("Displacement [mm]")
ax.set_ylabel("Force [kN]")
ax.legend()
ax.grid(True)
```

## 6. Practical Tips

- **Normalization**: When the force range varies across datasets, normalize the curves (e.g., divide by their maximum) before calling `prepare_data`, and undo the scaling after inference.
- **Augmentations**: For longer curves, random cropping or additive Gaussian jitter can augment the dataset before stacking the realizations.
- **Batching**: If you collect more than a few dozen repeats, set `n2n.batch_size` to keep inference memory-friendly.【F:src/autoden/algorithms/noise2noise.py†L118-L156】

Following this recipe lets you build a Noise2Noise denoiser tailored to eight replicates of a stress–strain experiment without needing clean ground-truth data.
