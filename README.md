# VoxScout: aerial person detector on an EfficientNet-B0 backbone

*Keywords: human detection, person detection, aerial, drone, search and rescue, TFLite, int8,
EfficientNet.*

**VoxScout** finds people in **aerial images** from a downward-looking camera. It takes a
160×160 RGB tile and marks where people are, so a search team can look where it matters.

It is built on **Google's EfficientNet-B0** (ImageNet weights from Keras, Apache 2.0): we
added a detection head and fine-tuned the whole network on HERIDAL. That is why it is a
separate project from [VoxHOD](https://github.com/ErdemWilkinson/VoxHOD), whose models are
built and trained from scratch, end to end. VoxScout uses the same data pipeline, training
code and evaluation as VoxHOD.

> **Türkçe özet:** VoxScout, drone görüntülerinde insan bulan bir model. Gövdesi Google'ın
> EfficientNet-B0'ı (ImageNet ağırlıkları, Apache 2.0); tespit başlığını ekleyip tüm ağı
> HERIDAL ile biz eğittik. Bu yüzden tamamen sıfırdan yapılan VoxHOD'dan ayrı bir proje.
> 4,4 milyon parametre, int8 dosyası 5,2 MB. VoxHOD'un aynı veriyle eğitilmiş nano
> tarifine göre aynı yakalama oranında 6-9 kat daha az yanlış alarm veriyor (float modelle
> ölçüldü). int8 dosyasının doğruluğu ve cihazdaki hızı henüz ölçülmedi. CC BY 4.0.

**Status:** float model measured on the HERIDAL test set, one WiSARD clip and 4,000 empty
NOMAD tiles. **Not measured yet:** the int8 file's accuracy and its speed on any device.
One training run. Weights: [Hugging Face](https://huggingface.co/inviolus/VoxScout).

## When to use it

- **VoxScout:** a ground station, a phone, a companion computer, or a second stage that
  checks only the few tiles a small model ranks highest.
- **[VoxHOD](https://github.com/ErdemWilkinson/VoxHOD) v3 / nano (197 / 75 KB):** a
  microcontroller that has to scan every tile. On a laptop CPU VoxScout takes about 0.4 s
  per tile (TFLite, 2 threads), so on an ESP32-P4 expect seconds per tile (estimate, not
  measured).

## Model

- **Input:** 160×160 RGB tile, scaled to `[-1, 1]` (`pixel / 127.5 - 1`). The model resizes
  it to 320 px inside.
- **Output:** a 40×40×3 grid (stride 4), same as VoxHOD. Channel 0 is the person-centre
  heatmap (sigmoid); channels 1 and 2 are `log(width)` and `log(height)` of the box in grid
  cells.
- **Architecture:** EfficientNet-B0 backbone; stride 8, 16 and 32 features merged into a
  CenterNet-style, anchor-free head. 4,355,142 parameters.
- **Detections:** local maxima of the heatmap (3×3 max-pool) above a threshold. Suggested
  threshold: **0.34** (chosen on validation).

| File | What it is |
|---|---|
| `voxscout_int8.tflite` | 5.2 MB, post-training quantized full int8; input `[1,160,160,3]` int8, output `[1,40,40,3]` int8. **Accuracy not measured** |
| `voxscout.h5` | Float model, the one measured below. Load with `tf.keras.models.load_model(path, compile=False)` |

```python
import numpy as np
import tensorflow as tf
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="voxscout_int8.tflite")
interpreter.allocate_tensors()
inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
(si, zi), (so, zo) = inp["quantization"], out["quantization"]

tile = np.asarray(Image.open("tile.png").convert("RGB").resize((160, 160)), np.float32) / 127.5 - 1
interpreter.set_tensor(inp["index"], np.clip(np.round(tile / si + zi), -128, 127).astype(np.int8)[None])
interpreter.invoke()
grid = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - zo) * so

heat = grid[..., 0]
peaks = heat == tf.nn.max_pool2d(heat[None, :, :, None], 3, 1, "SAME")[0, :, :, 0].numpy()
for y, x in zip(*np.nonzero(peaks & (heat >= 0.34))):
    w, h = np.exp(grid[y, x, 1:]) * 4
    print(f"person at ({(x + 0.5) * 4:.0f}, {(y + 0.5) * 4:.0f}) px, about {w:.0f}x{h:.0f}, score {heat[y, x]:.2f}")
```

## Training

- **Data:** [HERIDAL](https://zenodo.org/records/5662351) (CC BY 3.0), only the training part
  of VoxHOD's held-out-location split (`voxhod_g`, 55% of the tiles). It saw **less** data
  than the published VoxHOD v3 and nano.
- **Recipe:** VoxHOD's `train.py` with `VOX_ARCH=effb0`: 60 min on a laptop GPU, 19 epochs,
  batch 8, lr 5e-4 with cosine decay, last epoch kept. Same tiles, augmentation and loss as
  VoxHOD. Export: VoxHOD's `scripts/export_large.py`.

```
VOX_ARCH=effb0 VOX_DATA=D:/datasets/voxhod_g VOX_BATCH=8 VOX_LR=5e-4 VOX_MAX_MINUTES=60 python scripts/train.py
VOX_ARCH=effb0 VOX_ARTIFACTS=<run dir> python scripts/export_large.py
```

## Results (float model)

Threshold chosen on the held-out validation split (validation recall 78.1%), then scored on
test. The comparison columns are int8 models of VoxHOD nano and v3 size trained on the **same
split** (mean of 8 and 3 runs), so the data is equal. They are not the published VoxHOD files.

| Measure | VoxHOD nano recipe | VoxHOD v3 size | **VoxScout, float** |
|---|---:|---:|---:|
| HERIDAL test recall at 0.1 false alarms per tile | 66.0% | 66.0% | **84.1%** |
| HERIDAL test recall at 0.2 false alarms per tile | 74.7% | 74.6% | **87.1%** |
| HERIDAL test recall at 0.3 false alarms per tile | 77.6% | 77.3% | **88.2%** |
| HERIDAL test false alarms per tile at 78% recall | 0.32 | 0.33 | **0.036** |
| NOMAD false alarms per empty tile (unseen terrain) | 0.84 | 0.88 | **0.14** |
| WiSARD recall / false alarms per tile (other camera) | 58.1% / 0.29 | 50.5% / 0.33 | 53.6% / **0.03** |
| WiSARD recall on 8-16 px people | 30.7% | 31.8% | **22.6%** (worse) |

At threshold 0.34 VoxScout finds 79.8% of the HERIDAL test people with 67.0% precision
(WiSARD: 53.6% / 77.2%).

- **6 to 9 times fewer false alarms** than the VoxHOD recipes at the same recall.
- The gain comes from the ImageNet pretraining, not from size: a VoxHOD model of v3 size
  (2.6× nano) trained from scratch is no better than nano.
- **Weaker on the smallest people** on WiSARD, probably because the 160 px tile is upscaled.

**Not measured yet:**

- The int8 file's accuracy. A quick check on 40 test tiles with people: int8 and float flag
  the same 39 tiles at 0.34, heatmap correlation 0.91. Use the `.h5` if exact numbers matter.
- Speed and memory on any microcontroller.
- Repeat runs (one run), and training on the full HERIDAL training set.

## Limitations

- Trained only on HERIDAL: summer, daytime, mostly Mediterranean terrain. Snow, forest
  canopy, water, night and thermal are untested.
- One WiSARD clip and one NOMAD camera as external checks.
- Boxes are coarse (stride 4). The model says *where to look*.

## License

- **Weights** (`voxscout_int8.tflite`, `voxscout.h5`): **CC BY 4.0**, credit "VoxScout by Erdem
  Aslan" and HERIDAL. They contain a modified copy of Google's EfficientNet-B0 weights
  (Apache 2.0); see [MODEL_LICENSE.md](MODEL_LICENSE.md) and [NOTICE](NOTICE).
- **Code and documents in this repository:** [Apache License 2.0](LICENSE).
- Not trained on the Ferhat Tatık footage. NOMAD and WiSARD are test data only.

VoxScout is built for search and rescue: helping find people who are lost or injured. It is
not intended for targeting or harming people.
