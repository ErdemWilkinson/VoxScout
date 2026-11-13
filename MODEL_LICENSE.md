# Model and data licensing

The **VoxScout weights** (`voxscout_int8.tflite`, `voxscout.h5`) are released under **CC BY 4.0**.
Credit "VoxScout by Erdem Aslan" with a link to https://github.com/ErdemWilkinson/VoxScout, and HERIDAL.

## What the weights contain

| Part | Source | Licence |
|---|---|---|
| Backbone | EfficientNet-B0 ImageNet weights shipped with Keras (`keras.applications.EfficientNetB0`), Copyright Google LLC, **modified** (detection head added, whole network fine-tuned) | [Apache License 2.0](LICENSE) |
| Fine-tuning data | HERIDAL, Božić-Štulić et al., Zenodo record 5662351 | CC BY 3.0 |
| Test only, never trained on | WiSARD sample (Broyles, Hayner, Leung); NOMAD true-negative frames (ArtRuss/NOMAD, CC BY 4.0) | |

Google does not endorse VoxScout.

EfficientNet: M. Tan, Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks", ICML 2019.

HERIDAL: D. Božić-Štulić, Ž. Marušić, S. Gotovac, "Deep Learning Approach in Aerial Imagery for
Supporting Land Search and Rescue Missions", International Journal of Computer Vision, 2019.

WiSARD: D. Broyles, C. R. Hayner, K. Leung, "WiSARD: A Labeled Visual and Thermal Image Dataset for
Wilderness Search and Rescue", IROS 2022.

## ImageNet note

The backbone was pretrained by Google on ImageNet. ImageNet's images are distributed for
non-commercial research; whether those terms extend to weights trained on them is legally
unsettled. If you plan commercial use, check this for your case.

## Not used

Not trained on the Ferhat Tatık drone footage (licensed to the VoxHOD project for non-commercial
use only).

## Intended use

Search and rescue: helping find people who are lost or injured. Not intended for weapon targeting
or for harming people.
