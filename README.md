# WakeForge Studio

A local desktop workspace for building custom wake-word models for microcontrollers. Record a
phrase with your own microphone, expand the takes into an augmented dataset, train a compact
depthwise-separable CNN, quantize it to fully-INT8 TensorFlow Lite, and export a flash-ready C
header for ESP32-S3 (or another TFLite Micro target).

Everything runs on your machine. No audio, datasets, or models are uploaded anywhere.

![stack](https://img.shields.io/badge/Tauri_2-Rust_%2B_React-24C8DB)
![model](https://img.shields.io/badge/model-DS--CNN-8AE234)
![quantization](https://img.shields.io/badge/TFLite-INT8-FFB86C)

## Features

- Native microphone capture through `cpal`, with any device sample rate/format.
- Live waveform and level meter, 30-second capture limit, 10-second auto-stop.
- Energy-based VAD with silence trimming, pre/post roll, DC removal, and short fades.
- Per-phrase projects with recordings, datasets, models, and exports kept separate.
- Augmentation: pitch shift, time stretch, background noise at a chosen SNR, gain variation,
  uniform placement anywhere in the window, and balanced negatives that include other phrases,
  partial and reversed wake phrases.
- Real speech negatives: downloadable Russian speech corpora (hundreds of speakers) are cut into
  thousands of background windows, and an "Other speech" recorder mode captures your own voice
  saying anything but the wake phrase.
- Live test: the trained model on the PC microphone in real time, with the device's detector;
  every detection and every "it missed" moment is saved, can be played back and kept as a wake
  take or as other speech.
- Streaming test: hours of unseen speech run through the INT8 model like on the device, reporting
  false activations per hour against recall and exporting the chosen threshold for firmware.
- Optional AI helpers: speech-to-text reads the phrase back from your takes, a language model
  writes similar-sounding words and everyday phrases, and text-to-speech voices them (and your
  wake phrase) in many synthetic voices. Local models or any OpenAI-compatible API.
- Leakage-free source-level train/val/test split (all variants of one take stay together).
- Compact DS-CNN with log-Mel input and two output classes (`background`, `wake_word`).
- Fully-INT8 TFLite conversion verified against the test split.
- C export: model byte array, quantization/preprocessing config, exact Hann window and sparse
  Slaney mel filterbank, firmware guide, metadata, and training history.
- Real-time Python logs, epoch progress, and live accuracy/loss charts in the UI.
- Dark, minimalist single-workspace dashboard.

## Stack

| Layer | Technology |
| --- | --- |
| Desktop shell | Tauri 2, Rust 2021 |
| UI | React 18, Vite 6, TypeScript, Tailwind CSS, Radix primitives |
| Audio | `cpal`, Kaiser-windowed sinc resampler, energy VAD, `hound` WAV writer |
| ML pipeline | Python 3.10-3.12, TensorFlow 2.16-2.20, TensorFlow Lite, librosa, NumPy, soundfile |
| AI helpers (optional) | sherpa-onnx (STT, Supertonic TTS), piper-tts, Silero (PyTorch), llama.cpp server, OpenAI-compatible HTTP |
| Model export | Keras, TFLite converter, generated C11 headers |

## Requirements

| Tool | Version |
| --- | --- |
| Node.js | 20 or newer (npm included) |
| Rust | 1.77.2 or newer (`rustup`) |
| Python | **3.10, 3.11, or 3.12**, 64-bit |
| Microphone | Any input device visible to the OS |
| Disk | ~1 GB for TensorFlow, plus datasets and models |

TensorFlow does not currently publish mainstream desktop wheels for Python 3.13+. The app
probes the interpreter and reports an unsupported version instead of failing mid-training.

### Platform prerequisites

**Windows**

- Microsoft Visual C++ Build Tools (Desktop development with C++).
- WebView2 Runtime (included with current Windows 10/11).
- `py` launcher is convenient but not required.

**macOS**

- Xcode Command Line Tools: `xcode-select --install`.

**Linux (Debian/Ubuntu)**

```bash
sudo apt update
sudo apt install build-essential curl file libssl-dev libwebkit2gtk-4.1-dev \
  libayatana-appindicator3-dev librsvg2-dev libasound2-dev libxdo-dev pkg-config wget
```

Install Rust from <https://rustup.rs> if `cargo` is not already available.

## Quick start

```bash
npm install
```

Set up the Python environment. The scripts create `.venv` in the project root, install the ML
requirements, and verify the packages.

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

**macOS / Linux:**

```bash
bash scripts/setup.sh
```

Then start the desktop app:

```bash
npm run tauri dev
```

A plain `npm run dev` only serves the React UI in a browser. The browser cannot call Tauri
commands, so recording and training require `npm run tauri dev` (or a packaged build).

### Setup script options

| Option | Windows | macOS/Linux | Meaning |
| --- | --- | --- | --- |
| Recreate venv | `-Recreate` | `--recreate` | Delete `.venv` and rebuild it |
| Skip npm | `-SkipNode` | `--skip-node` | Only set up Python |
| Custom Python | `-PythonExe <path>` | `PYTHON=<path> bash scripts/setup.sh` | Use a specific interpreter |

The app looks for Python in this order:

1. `WAKE_WORD_PYTHON` environment variable (path or command).
2. `<project>/.venv/Scripts/python.exe` or `<project>/.venv/bin/python`.
3. `<project>/python/.venv/...`.
4. `.venv` inside the bundled resource directory.
5. `python` / `python3` on `PATH`.

A release bundle includes `python/` scripts but not `.venv`. End users must install Python and
the requirements, or point `WAKE_WORD_PYTHON` at a prepared interpreter. A future installer can
ship a prebuilt environment.

## Workflow

### 1. Create or pick a project

Projects are listed in the sidebar, most recently active first. **+** creates one: enter the
wake phrase exactly as it will be spoken (for example `hey jarvis` or `Среда`). The dialog shows
the folder name (slug) as you type and rejects a phrase or slug another project already uses.
Latin and Cyrillic are transliterated (`Среда` → `sreda`); other scripts get a short hash suffix,
so different phrases never share a folder.

The **wake phrase bar** above the steps holds what you actually say to the device and its
language. It starts as the project name and can differ from it (a project called "kitchen"
can listen for «Среда»). Every step uses it: other-speech prompts, word lists and synthetic
voices, the dataset (speech-corpus utterances that contain it are skipped), model metadata and
the streaming test. *Check takes* runs speech-to-text over every take, shows the most common
transcript and how many takes agree, and lists the takes that disagree (worth re-recording).

- **Rename** (pencil) changes the phrase and moves the folder to the new slug. Takes, dataset
  and model move with it; retrain to update the phrase in the exported headers. If the spoken
  phrase changes, create a new project instead.
- **Delete** (bin) permanently removes the project folder. When it holds takes, a dataset or a
  model, you have to type the phrase to confirm. Packages exported elsewhere are not touched.
- Switching, renaming and deleting are blocked while a take is being recorded or a dataset or
  training job runs. The backend enforces the same rule for rename and delete.

Each project is fully isolated: its own `recordings/`, `dataset/`, `models/` and `exports/`. A
take is always saved into the project it was started in, pipeline events carry their project,
and every panel is reset when you switch, so logs or charts from one project never show up in
another. The app remembers the last opened project.

### 2. Record clean takes

- Speak the exact phrase naturally, with a short pause before and after.
- Record **20 or more takes** if possible. Three is the hard minimum for a leak-free split, and
  recall measurements need several held-out takes to mean anything.
- Vary distance, pitch, speed, and room position between takes. Do not add background noise
  manually; augmentation does that.
- The recorder downmixes to mono, resamples to 16 kHz with a band-limited windowed-sinc filter
  (flat to ~7.5 kHz, >50 dB alias rejection), removes DC offset, detects speech energy, trims
  silence, and adds 160 ms of pre/post roll.
- Takes are stored as 16-bit PCM WAV, 16 kHz mono. You can play, rename, or delete them in the UI.

The VAD uses 20 ms frames, estimates the noise floor from the 20th percentile, and requires at
least 250 ms of detected speech. A take that is too quiet or has no speech is rejected with a
message instead of being saved.

**Other speech.** Switch the recorder to *Other speech* and read the prompts aloud. Prompts
interleave similar-sounding words (three per prompt) and everyday phrases from the Phrases word
lists, which are written for your wake phrase, with built-in general sentences (40 Russian, 20
English) that are not tied to any phrase. Without word lists only the general sentences are
shown, and the prompt card offers *Generate similar words* when a language model is set up. Takes
are stored in `negatives/` with the prompt text and are always used as hard negatives in the
next dataset build. A model trained mostly on your voice saying the wake phrase otherwise learns
"this voice = wake word"; 20-40 such takes (about three minutes) are the cheapest fix.

### 3. Phrases: AI-generated negatives (optional)

The weak spot of a personal wake-word model is negatives: it rarely hears other words in your
voice or in similar voices. The **Phrases** panel automates that:

1. **Wake phrase.** Lists and voices are made for the phrase and language in the wake phrase
   bar at the top.
2. **Negative word lists.** *Generate* asks the language model for three groups: similar-sounding
   words, phrases sharing a part of the wake phrase, and everyday speech. Answers are filtered:
   duplicates, items in the wrong script and anything that *contains the whole wake phrase*
   (which would legitimately trigger a sliding-window detector) are dropped. Every list stays
   editable; you can type your own words too.
3. **Synthetic voices.** Tick one or more TTS engines under *Voices from* (for Russian:
   Supertonic 3 with 10 voices, Silero with 5, Piper with 4). *Synthesize* voices every word in
   1-5 voices in total, shared out over the ticked engines (with three engines and 2 voices per
   word each word is still voiced twice), at varied speed, and optionally the wake phrase itself
   in every voice of every engine (up to 40 per engine) at three speeds. Counting voices per
   engine instead tripled the similar-word negatives with three engines and cut recall at one
   false activation per hour from 81% to 69% on the «Среда» test; more voices for the wake phrase
   itself did not hurt. Clips are cached per engine, text, voice
   and speed in `<project>/synthetic/engines/<engine>/`: a later run synthesizes only new words,
   voices or engines, and an engine whose clips are all cached is not even loaded. Voices for a
   word are picked from a seed of the engine and the word, so raising *voices per word* or
   adding words keeps the existing clips. `synthetic/manifest.json`, which the dataset build
   reads, lists only clips of the ticked engines for the current phrase and word lists; clips
   of unticked engines stay cached on disk for later.

The next dataset build uses them automatically (a checkbox in the dataset panel turns them off):
synthetic wake-phrase clips go to **train only**, so validation and test still measure your real
recordings; synthetic negatives are split across train/val/test **by text**, so one word never
appears on both sides.

#### Providers

**AI models** (header button, or the chips in the Phrases panel) has one tab per category:
speech-to-text, word lists (LLM) and text-to-speech. Exactly one provider is active per
category. Each local model shows its languages, download size, RAM and license, and is marked
*recommended* or *no <language>* for the project's language, so it is visible up front whether a
model can handle your phrase.

| Category | Local models (download on demand) | Languages |
| --- | --- | --- |
| STT | GigaAM v3 RNN-T (Sber) | Russian |
| STT | Parakeet TDT 0.6B v3 (NVIDIA), int8 | 25 European languages |
| STT | Whisper large-v3 turbo | ~99 languages |
| LLM | Qwen3.5 4B / 2B (Q4_K_M), Gemma 4 E4B, via a managed llama.cpp server (Vulkan build, uses the GPU when present) | multilingual |
| TTS | Supertonic 3, 10 voices | 31 languages |
| TTS | Silero v5, PyTorch CPU | Russian |
| TTS | Piper (Denis, Dmitri, Irina, Ruslan) | Russian |
| TTS | Piper LibriTTS-R, 904 speakers | English |

Models are stored in `%APPDATA%\com.wakeforge.studio\models` (other platforms: the app data
folder). Python packages a model needs (`sherpa-onnx`, `piper-tts`, CPU `torch`) are installed
into the app's Python environment with one click when missing.

**API endpoints** cover everything else: any OpenAI-compatible server (OpenAI, OpenRouter, Groq,
local Ollama / LM Studio). One endpoint can serve several categories (`/chat/completions`,
`/audio/speech`, `/audio/transcriptions`); set a model per category, optional TTS voice names and
the languages it supports (leave empty if unknown; the UI then says "language not declared").
API keys are kept in the OS credential store (Windows Credential Manager, macOS Keychain; on
Linux the kernel keyring, which forgets keys after a reboot) and reach the Python job only
through its environment. *Test* runs a tiny request against the active
provider of a category.

New providers plug in at `python/voicekit/`: implement `SttEngine`, `TtsEngine` or `LlmEngine`
(`base.py`), add catalog entries (`catalog.py`) or a profile type, and map it in
`registry.py`. New word groups are a prompt in `groups.py`.

### 4. Build the dataset

Choose:

- **Variations per take**: 10-40, default 30. On a 10-take test, 30 variations raised wake-word
  recall from ~93% to ~99% compared with 12.
- **Window**: 1.0, 1.5, 2.0, or 2.5 seconds, default 1.5 s.
- **Backgrounds** (optional): a folder of WAV/FLAC/OGG files with room tone, music, fans, traffic,
  or other non-wake-word audio.
- **Negative phrases** (optional): a folder of recordings of other words and phrases, ideally ones
  that sound similar to the wake phrase. They are augmented exactly like the wake word, so the
  model has to learn the words, not "speech vs. no speech". Files are split by source like the
  takes; with fewer than three files they are used for training only.

Augmentation applies, per variant:

- Pitch shift up to ±2.5 semitones.
- Time stretch 0.88-1.12×.
- Noise mixed at 5-25 dB SNR over the **complete window** (real backgrounds when provided,
  otherwise synthetic white/pink/brown/hum/click noise).
- Peak normalization to 0.2 (-14 dBFS) followed by a -30 to +6 dB gain range, so speech is seen
  from about -44 to -8 dBFS peak, like raw microphone input from across a room up to close range.
  With the earlier -12 dB floor, cross-validation recall fell from 96% to 14% for takes quieter
  than -25 dBFS; the wider range fixes that at the cost of roughly twice the false activations
  at a given threshold (`--min-gain-db` sets the floor; measure the real input level on the
  device before tightening it).
- Uniform placement anywhere the phrase fits completely (20 ms from the edges), because a sliding
  window on the device sees the phrase at every alignment. Edge fades.
- Room reverb on 35% of variants: a synthetic impulse response (RT60 0.15-0.7 s, direct-to-reverberant
  ratio -2 to +12 dB), because the device hears you across a room while takes are recorded close up.
- Other speech around the phrase on 30% of variants (when speech corpora are installed): corpus
  speech of the same split fills the window before and after the phrase, 40-250 ms apart and
  2-12 dB quieter, as in "so, <phrase>, turn on the light".

Reverb and surrounding speech are applied to speech negatives (other phrases, partial and reversed
phrases) exactly as to positives, so neither becomes a shortcut. `manifest.csv` records both per clip.

Takes that are longer than 97% of the window after trimming are skipped with a warning instead
of being cut and still labelled as a complete wake word; the dataset panel warns before the build.
Time stretching never slows a phrase down so far that it stops fitting the window.

Negatives balance each split one-to-one with positives:

| Kind | Share | Source |
| --- | --- | --- |
| Other phrases | 35% | negative phrase folder, if provided |
| Real backgrounds | 30% | background folder, if provided |
| Partial wake phrase | 10% | first or last 25-60% of a take from the same split |
| Reversed wake phrase | 5% | a take from the same split, time-reversed |
| Synthetic | rest | digital silence, dither, room tone, hum, clicks, noise, tones |

Missing sources hand their share to the synthetic kinds. `dataset_info.json` records the per-split
mix under `negative_kinds`, and `manifest.csv` has a `kind` column.

**Real speech** (*AI models → Speech data*) is added on top of the balanced negatives: 2,000 to
20,000 windows (default 6,000) cut from the installed training corpora, split train/val/test by
utterance, with noise mixed in like every other clip. Utterances whose transcript contains the
wake phrase are skipped. This is the single most effective defence against "fires on any
speech":

| Model (wake word «Среда», 10 takes) | False activations / hour at 0.5 | At ~1 / hour: recall |
| --- | --- | --- |
| Takes + synthetic negatives only | ~1,700 | not reachable |
| + 3 h of Golos crowd speech | ~110 | 81% |
| + 12 h of Golos crowd and far-field speech | ~170 | 88% |

Measured with the streaming test on 4.4 h of FLEURS and Golos far-field audio never used for
training; recall over the four held-out takes in 48 contexts.

| Corpus | Role | Hours | Notes |
| --- | --- | --- | --- |
| Golos crowd (SberDevices) | training negatives | 9.0 | phone recordings of assistant requests |
| Golos far-field (SberDevices) | training negatives | 3.4 | smart-speaker requests across a room |
| FLEURS Russian test | benchmark | 2.5 | read Wikipedia sentences, CC BY 4.0 |
| Golos far-field test | benchmark | 1.4 | held-out far-field requests |

**Hard negative mining.** After a streaming test, *Mine* in the training panel streams the
current model over the training part of the speech corpora and saves up to 1,500 windows that
score 0.3 or more (one per second of an utterance, loudest first) into the project's `mined/`
folder. The next dataset build adds them to the training negatives only; rebuild, retrain, test
and mine again. Corpus utterances are assigned to train/val/test by a hash of their id, so every
tool agrees on the split and mined windows never come from validation or test speech.

Corpora are converted once to 16 kHz int16 (`audio.pcm`) plus `index.json` with every
transcript, in `<app data>/datasets/<id>/`, and shared by all projects. Common Voice moved to
the Mozilla Data Collective and needs an account, so it is not in the catalog.

The split is **by source take**, not by augmented clip. The default target is 70% train,
15% validation, 15% test, with at least one take forced into each split. With only three takes
that means one take per split, so validation/test metrics are very coarse. Five to ten takes are
strongly recommended.

Rebuilding a dataset replaces the previous augmented dataset but leaves clean recordings and
trained models untouched. A trained model is not automatically invalidated when the dataset
changes; retrain after rebuilding if you want the metrics to match.

### 5. Train

Defaults in the UI:

| Parameter | Default |
| --- | --- |
| Epochs | 40 (maximum; early stopping may finish sooner) |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Early-stopping patience | 10 epochs, not before epoch 10 |
| Model selection | lowest validation loss |

The training script:

1. Loads each split, resamples if needed, and computes log-Mel features.
2. Standardizes features with the **training-set** mean and standard deviation.
3. Applies balanced class weights.
4. Trains the DS-CNN with early stopping and learning-rate reduction.
5. Evaluates the float model on the held-out test split.
6. Converts to fully-INT8 TFLite (`TFLITE_BUILTINS_INT8`, int8 input and output).
7. Re-evaluates the quantized model with the same test split.
8. Writes C headers, metadata, history, and the preprocessing guide.

Validation loss selects the restored weights because the small validation split saturates at
100% accuracy early and then cannot rank epochs. `history.json` reports the epoch whose weights
were actually exported (`best_epoch`, `weights_restored`). With 10 or fewer epochs the final
weights are used.

All artifacts are written to a hidden staging folder next to `models/` and swapped in only after
the whole export succeeded. A failed or canceled run leaves the previous model intact, so the
`.tflite` file and the quantization constants in the headers always belong together.

Live logs and per-epoch metrics stream into the UI. Only one heavy job runs at a time; recording
is blocked while a job runs, and jobs are blocked while recording.

**Streaming test.** Test-split accuracy is measured on clips that are half noise and silence, so it
says little about a detector that runs ten times per second on live audio. *Run test* in the
training panel joins every installed benchmark corpus into one stream and scores it with the
INT8 model every 100 ms, exactly as firmware would. Two detector rules are compared: *k in a row*
fires when each of the last *k* windows scores at or above the threshold, *avg of k* when their
mean does (it tolerates one uncertain window). Each detection is followed by a 1.5 s refractory
period. The held-out
takes are also embedded in quiet, noisy and speech contexts at random levels to measure recall.
The panel shows false activations per hour and recall for thresholds 0.5-0.99 and both rules,
picks the best recall at no more than one false activation per hour, lists the utterances the
model confuses with the phrase, shows recall per held-out take (weakest first, by context), and
writes `evaluation.json` and `wake_word_detector.h`. A take the model never catches is usually a
defective recording (on the «Среда» project: a cut-off or missing initial «с») and is worth
re-recording or deleting; *Check takes* in the wake phrase bar finds the same takes by
speech-to-text.

For stable comparisons `augment.py --folds N --fold K` holds out fold K of N (by a fixed shuffle
of take names) instead of a random split; running every fold tests every take once. With only a
handful of held-out takes, which takes land in validation moves recall by 15+ points.

### 6. Export

Click **Export for ESP32** and choose a destination folder. The package contains:

| File | Purpose |
| --- | --- |
| `wake_word_model.h` | `const uint8_t model_data[]`, length, 16-byte alignment |
| `wake_word_config.h` | Sample rate, window/hop/FFT/mel constants, int8 quantization, feature mean/std |
| `wake_word_mel.h` | Exact periodic Hann window and sparse Slaney mel filterbank |
| `PREPROCESSING.md` | Step-by-step firmware frontend and a C sketch |
| `wake_word_int8.tflite` | Fully quantized TFLite graph |
| `metadata.json` | Model, metrics, quantization, training, artifact manifest |
| `history.json` | Per-epoch accuracy/loss and best epoch |
| `export_info.json` | Compact manifest for firmware builds or CI |
| `wake_word_detector.h` | Threshold, windows in a row, step and refractory time from the streaming test |
| `evaluation.json` | Full streaming-test results |

The `.keras` model stays in the project `models/` directory for inspection but is not copied
into the export package.

### 7. Live test

*Start listening* in step 06 runs `python/live.py` on the default PC microphone (captured at the
device rate and resampled to 16 kHz with soxr). Every 100 ms the last 1.5 s go through the INT8
model and the same detector rule as the streaming test and the ESP32 firmware; the rule, windows
and threshold start from the setting the streaming test picked. The panel shows the score, a 10 s
score graph with the threshold, the input level against the range the model was trained on, and
a list of events:

- **detection**: 2 s before to 0.6 s after the detection is saved;
- **missed**: press *It missed* right after saying the phrase without a reaction; the last 4 s are
  saved.

Play any clip, then keep it: *Wake take* adds it to the takes (for a detection only 1.5 s before
to 0.4 s after the detection, trimmed by the recorder's VAD, so earlier speech does not end up in
a positive), *Other speech* adds it to the other-speech takes as a hard negative. Rebuild the
dataset and retrain to use them. The live test is a project job: recording, other jobs and
switching projects wait until it is stopped. The `sounddevice` package is installed on first use.

## Signal and model details

### Log-Mel frontend

| Parameter | Value |
| --- | --- |
| Sample rate | 16,000 Hz, mono |
| Analysis window | 480 samples (30 ms), periodic Hann |
| FFT size | 512 |
| Window placement | 16 zeros on each side (centered in the FFT buffer) |
| Hop | 320 samples (20 ms) |
| Mel filters | 40, Slaney scale and Slaney area normalization |
| Mel range | 20-7,600 Hz |
| Log floor | `log(mel + 1e-6)` |
| Frames | `1 + (clip_samples - 512) / 320` |

For the default 1.5 s window: 24,000 samples and **74 frames**. Input shape is
`[1, 74, 40, 1]` int8 after quantization, with per-tensor input scale/zero-point stored in
`wake_word_config.h`.

Training uses `librosa.feature.melspectrogram(..., center=False, power=2.0)`. The generated
`wake_word_mel.h` reproduces the same window and filters for firmware, so no dense mel table is
needed. The desktop recipe was cross-checked against the firmware recipe to below `1e-6`
log-Mel error and zero int8 mismatches on the test data.

### DS-CNN

```
Input [frames, 40, 1]
  Conv 3x3, 32 filters, stride 2, BN, ReLU6
  DS block 1: depthwise 3x3 (64) + pointwise 1x1 (64), stride 1
  DS block 2: depthwise 3x3 (64) + pointwise 1x1 (64), stride 2
  DS block 3: depthwise 3x3 (96) + pointwise 1x1 (96), stride 1
  DS block 4: depthwise 3x3 (96) + pointwise 1x1 (96), stride 2
  Global average pool
  Dropout 0.2
  Dense 2, softmax
```

Each DS block uses BatchNorm and ReLU6. BatchNorm momentum is `0.2` (instead of Keras' default
`0.99`) because small datasets have few updates per epoch; with the default, validation can look
stuck at 50% while training accuracy rises. After early stopping restores the best weights,
`train.py` runs one extra forward pass over the training set and replaces all BN moving
statistics with the measured dataset moments. The exported TFLite model therefore uses correct
inference-mode normalization. Early stopping starts at epoch 10 for the same reason.

The architecture has **26,722 parameters** and produces roughly **48 KiB** of TFLite weights for
the default 1.5 s window; exact file size varies slightly with quantization ranges.

**Temporal head (default).** *Classifier head → Temporal* replaces the global average pool with
a frequency pool, two time convolutions (~0.75 s of ordered context) and a max over time, so the
order of sounds matters (~100k parameters, ~100 KB int8; trivial for an ESP32-S3). It pays off
together with the reverb and surrounding-speech augmentation: on the «Среда» test (23 takes,
6 held out, 4.4 h of unseen speech) recall at about one false activation per hour went from 76%
(average pool) to 92%, while the average-pool head got worse with the same data (72%), because
averaging the whole window cannot separate the phrase from the speech around it. *Average pool*
(~27k parameters) remains available for very small targets.

### Quantization

The converter uses post-training full-integer quantization with a deterministic, class-balanced
representative sample (up to 400 training features). Inputs and outputs are int8, and only
built-in int8 TFLite ops are allowed so TensorFlow Lite Micro does not need float fallback
kernels. The quantized model is evaluated on the test split; both float and int8 metrics are
shown in the UI and stored in `metadata.json`.

## Firmware integration

Copy `wake_word_model.h`, `wake_word_config.h`, and `wake_word_mel.h` into your firmware source
tree:

```c
#include "wake_word_model.h"
#include "wake_word_config.h"
#include "wake_word_mel.h"

const uint8_t *model = model_data;
const size_t model_len = model_data_len;
```

Use the TFLite Micro `MicroInterpreter` with the model ops resolver, then feed one full int8
window (`WW_FRAME_COUNT * WW_MEL_BINS` bytes, row-major time × mel). Class index 1 is the wake
word. Dequantize the two output scores with `WW_OUTPUT_SCALE` and
`WW_OUTPUT_ZERO_POINT`.

**Tensor arena:** `metadata.json` recommends **128 KiB** as a starting point. This is an
estimate, not a guarantee. Measure `interpreter.arena_used_bytes()` on the target and adjust.
The stride-2 blocks reduce activation memory, but your compiler, TFLM version, and op set can
change the peak.

**FFT scaling:** the firmware frontend must use an unscaled 512-point real FFT. If your FFT
scales the forward transform by `1/N`, set `WW_FFT_POWER_SCALE` to `N*N` or multiply the power
bins by `N*N` before the mel filters. `PREPROCESSING.md` contains the exact frame loop.

**Detection:** the model classifies a fixed window. For live use, slide it every
`WW_DETECT_STEP_MS` and keep the last `WW_DETECT_WINDOWS` wake scores. With
`WW_DETECT_AVERAGE` 1, fire when their mean reaches `WW_DETECT_THRESHOLD`; with 0, when each of
them does (all values come from `wake_word_detector.h`, written by the streaming test). Then
ignore detections for `WW_DETECT_REFRACTORY_MS`. Re-check the threshold on recordings from the real
device and microphone. Do not ship the test-split accuracy as a field accuracy claim.

## Project layout

Application data (the UI footer shows the exact path):

- Windows: `%APPDATA%\com.wakeforge.studio\projects\<slug>`
- macOS: `~/Library/Application Support/com.wakeforge.studio/projects/<slug>`
- Linux: `~/.local/share/com.wakeforge.studio/projects/<slug>`

```
<slug>/
  project.json
  recordings/
    rec_YYYYMMDD_HHMMSS_xxxxx.wav
    rec_YYYYMMDD_HHMMSS_xxxxx.json    # sidecar metadata
  negatives/                          # "other speech" takes + prompt text
  mined/                              # hard negative windows + manifest.json (optional)
  live/                               # clips of the last live test
  dataset/
    train/wake_word/*.wav
    train/background/*.wav
    val/...
    test/...
    manifest.csv
    dataset_info.json
  voice.json                          # spoken phrase, language, word lists
  synthetic/                          # TTS clips (optional)
    manifest.json                     # clips of the ticked engines, read by the dataset build
    engines/<engine>/*.wav            # per-engine cache + cache.json
  models/
    wake_word.keras
    wake_word_int8.tflite
    wake_word_model.h
    wake_word_config.h
    wake_word_mel.h
    PREPROCESSING.md
    metadata.json
    history.json
    export_info.json
    evaluation.json                   # streaming test (optional)
    wake_word_detector.h
  exports/                            # optional local copies
```

Shared speech corpora live next to the projects in `datasets/<id>/` (`audio.pcm`, `index.json`).

The repository layout:

```
src/                     React UI, hooks, components
src-tauri/src/
  lib.rs                 Tauri commands and app state
  audio.rs               cpal capture, resampling, VAD
  project.rs             project layout, recording and export files
  python_bridge.rs       Python discovery, process jobs, events
  voice.rs               AI provider settings, API keys, project voice data
python/
  check_env.py           dependency probe used by the UI
  augment.py             dataset builder
  train.py               training, quantization, C headers
  evaluate.py            streaming test: false activations per hour, recall, detector header
  live.py                live test on the PC microphone
  mine.py                hard negative mining over the training speech
  voice.py               AI helper jobs: status, install, transcribe, words, synthesize, corpus
  voicekit/              engine interfaces, model and speech corpus catalogs, providers, word groups
  requirements.txt
scripts/
  setup.ps1              Windows setup
  setup.sh               macOS/Linux setup
```

## Development

Useful commands:

```bash
npm run check                                  # TypeScript type-check
npm run build                                  # production frontend build
npm run tauri dev                              # desktop app with hot reload
npm run tauri build                            # packaged desktop app

cargo check --manifest-path src-tauri/Cargo.toml
.venv/Scripts/python.exe -m py_compile python/augment.py python/train.py python/check_env.py
```

The Python scripts can also be run directly, which is useful for debugging:

```bash
.venv/Scripts/python.exe python/augment.py \
  --input "path/to/recordings" --output "path/to/dataset" \
  --phrase "hey jarvis" --variations 30 --clip-seconds 1.5 --overwrite

.venv/Scripts/python.exe python/train.py \
  --dataset "path/to/dataset" --output "path/to/models" \
  --phrase "hey jarvis" --epochs 40 --batch-size 64 --learning-rate 0.001
```

Progress is emitted as `@@PROGRESS@@ {json}` and the final payload as `@@RESULT@@ {json}`.
Human-readable lines are ordinary stdout/stderr. The Rust bridge parses the markers, strips them
from UI logs, and forwards `job-log`, `job-progress`, and `job-finished` events.

## Troubleshooting

**The UI says the ML environment needs setup.** Open the setup panel; it shows the interpreter,
version, missing packages, and script directory. Run the setup script with `-Recreate` /
`--recreate` if the `.venv` was made with Python 3.13 or is broken.

**`npm run tauri dev` opens a blank window.** Make sure Vite is running on port 1420 and that the
CSP in `src-tauri/tauri.conf.json` still contains
`connect-src 'self' ipc: http://ipc.localhost`. Check the terminal for Rust or frontend errors.

**No microphone / no audio captured.** Confirm the default input device in the OS, close other
apps that may hold exclusive access, and check the input level. The app uses the system default
input; switch it in the OS settings if needed.

**"No speech detected above the noise floor."** Move closer, reduce background noise, or speak
longer. The VAD requires at least 250 ms of speech and a signal peak above 0.004.

**Training fails with a Python traceback.** Read the live log panel. The most common causes are
missing TensorFlow, Python 3.13, or an empty train/val/test folder. `train.py` requires at least
one usable clip in each split.

**Test accuracy is 100% but the device misses the word.** The test split is tiny and comes from
the same takes and rooms as training. Record more speakers, distances, and rooms; add real
backgrounds and similar-sounding negative phrases; test on the target microphone.

**TFLite Micro runs out of arena.** Increase the arena, measure `arena_used_bytes()`, and check
that the model is using built-in int8 kernels. The 128 KiB recommendation is only a starting
point.

**The model header is huge in the editor.** It is a byte array; keep it in flash (`const`) and
exclude generated headers from clang-format or language-server indexing if needed.

## Limitations

- No live inference or microphone testing on the MCU; this tool produces the model and frontend
  constants, not a complete firmware application.
- Without speech corpora or other-speech takes, speech negatives are limited to partial and
  reversed wake phrases and the model fires on almost any voice. Download the speech data.
- The speech corpus catalog is Russian only for now; other languages need a corpus entry in
  `python/voicekit/corpora.py`.
- The source-level split gives very small validation/test sets when only 3-4 takes are recorded.
- Real background files are currently shared across splits. This is less severe than wake-word
  leakage, but a source-level background split would give stricter metrics.
- The streaming test simulates the firmware detector on the PC; it does not replace a test on
  the device with its own microphone.
- Only one dataset/training job at a time. There is no bulk import UI yet.
- Accuracy numbers are dataset-specific and are not a substitute for field testing.
- The exported package targets TFLite Micro. Other runtimes (ONNX, ESP-DL, CMSIS-NN) would need
  separate conversion and kernels.

## License

No license file is included in this repository. Add one before distributing the app or generated
models.
