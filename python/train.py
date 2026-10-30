#!/usr/bin/env python3
"""Train a compact DS-CNN wake-word classifier and export INT8 TFLite + C headers.

Design goals:
  * fixed 16 kHz mono input window (default 1.5 s)
  * log-Mel features computed with plain STFT parameters that can be replicated
    on an MCU
  * a compact depthwise-separable CNN that fits an ESP32-S3 flash budget
  * fully-int8 TFLite model (no float fallback kernels)
  * one C array with the model and a second C header with preprocessing metadata

The backend streams ``@@PROGRESS@@ {json}`` lines and emits the final
``@@RESULT@@ {json}`` line. Human-readable messages are normal stdout/stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("librosa is missing. Run: pip install -r python/requirements.txt") from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is missing. Use Python 3.10-3.12 and run: "
        "pip install -r python/requirements.txt"
    ) from exc

SAMPLE_RATE = 16_000
LABELS = ["background", "wake_word"]
WAKE_CLASS = 1
PROGRESS_PREFIX = "@@PROGRESS@@"
RESULT_PREFIX = "@@RESULT@@"

# These values are mirrored in wake_word_config.h for firmware authors.
WIN_LENGTH = 480  # 30 ms
HOP_LENGTH = 320  # 20 ms
N_FFT = 512
WINDOW_OFFSET = (N_FFT - WIN_LENGTH) // 2  # librosa centers shorter windows in the FFT
N_MELS = 40
FMIN = 20.0
FMAX = 7600.0

# Default Keras BN momentum (0.99) needs ~100 updates before the moving
# statistics are usable. Small wake-word datasets often have only a few
# batches per epoch, so validation looks stuck at 50% while training accuracy
# climbs. A faster moving average keeps validation (and the exported model)
# representative much earlier.
BN_MOMENTUM = 0.20
# Early stopping restores the best epoch's weights, including BN moving
# statistics. A final explicit recalibration pass on the training set removes
# any remaining bias in those statistics (and the exported INT8 model inherits
# correct inference-mode normalization).
BN_RECALIBRATION_SAMPLES = 1024
MIN_EPOCHS_BEFORE_EARLY_STOP = 10
# The validation split is small, so val_accuracy saturates at 1.0 quickly and
# can no longer tell epochs apart. val_loss keeps ranking them by confidence.
EARLY_STOP_MONITOR = "val_loss"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def emit_progress(stage: str, progress: float, message: str, **extra: object) -> None:
    print(
        f"{PROGRESS_PREFIX} "
        + json.dumps(
            {
                "stage": stage,
                "progress": max(0.0, min(1.0, float(progress))),
                "message": message,
                **extra,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def emit_result(payload: dict) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def safe_float(value: object, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def wavs_in(*folders: Path) -> list[Path]:
    found: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            found.extend(
                p
                for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg"}
            )
    return found


def fit_audio(audio: np.ndarray, clip_samples: int) -> np.ndarray:
    out = np.zeros(clip_samples, dtype=np.float32)
    if audio.size == 0:
        return out
    if audio.size >= clip_samples:
        start = (audio.size - clip_samples) // 2
        out[:] = audio[start : start + clip_samples]
    else:
        start = (clip_samples - audio.size) // 2
        out[start : start + audio.size] = audio
    return out


def extract_logmel(audio: np.ndarray, clip_samples: int) -> np.ndarray:
    """Return a [frames, mel] log-Mel patch without external normalization."""
    audio = fit_audio(np.asarray(audio, dtype=np.float32), clip_samples)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        window="hann",
        center=False,
        power=2.0,
    )
    return np.log(mel + 1e-6).T.astype(np.float32)


def load_split(dataset_dir: Path, split: str, clip_samples: int) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    expected_frames = 1 + (clip_samples - N_FFT) // HOP_LENGTH
    for label_index, label_name in enumerate(LABELS):
        folder = dataset_dir / split / label_name
        files = wavs_in(folder)
        if not files:
            log(f"WARNING: no {label_name} files in {split}")
            continue
        for path in files:
            try:
                audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="kaiser_best"
                    )
                feature = extract_logmel(audio, clip_samples)
                if feature.shape[0] != expected_frames:
                    log(
                        f"WARNING: {path.name} produced {feature.shape[0]} frames, "
                        f"expected {expected_frames}; skipping"
                    )
                    continue
                features.append(feature)
                labels.append(label_index)
            except Exception as exc:
                log(f"WARNING: could not load {path}: {exc}")
    if not features:
        raise ValueError(f"No usable audio in split '{split}'.")
    x = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    log(f"{split}: {len(y)} samples, feature shape {x.shape[1:]}")
    return x, y


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x[..., None], y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 4096), seed=1337, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def ds_block(x: tf.Tensor, filters: int, name: str, stride: int = 1) -> tf.Tensor:
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        strides=(stride, stride),
        padding="same",
        depth_multiplier=1,
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_dw_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_dw_relu")(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding="same", use_bias=False, name=f"{name}_pw"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_pw_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_pw_relu")(x)
    return x


ARCHITECTURES = {
    "ds-cnn": "DS-CNN",
    "ds-cnn-temporal": "DS-CNN + temporal head",
}


def temporal_head(x: tf.Tensor) -> tf.Tensor:
    """Keep the time axis: pool frequency, convolve over time, then max over time.

    Global average pooling throws away the order of sounds, so "сре-да" and a
    word that merely contains similar pieces look alike. Here the last feature
    map is averaged over frequency only, two time convolutions see ~0.75 s of
    ordered context, and a max over time keeps the detector independent of
    where the phrase sits in the window. All ops are standard TFLM kernels
    (AVERAGE_POOL_2D, CONV_2D, REDUCE_MAX).
    """
    x = tf.keras.layers.AveragePooling2D(pool_size=(1, x.shape[2]), name="freq_pool")(x)
    for index, kernel in enumerate((5, 3), start=1):
        x = tf.keras.layers.Conv2D(
            96, (kernel, 1), padding="same", use_bias=False, name=f"time{index}"
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"time{index}_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"time{index}_relu")(x)
    return tf.keras.layers.GlobalMaxPooling2D(name="pool")(x)


def build_model(input_shape: tuple[int, int, int], architecture: str = "ds-cnn") -> tf.keras.Model:
    """DS-CNN variant sized for ESP32-S3 / Cortex-M class targets."""
    inputs = tf.keras.Input(shape=input_shape, name="logmel")
    x = tf.keras.layers.Conv2D(
        32, (3, 3), strides=(2, 2), padding="same", use_bias=False, name="stem"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name="stem_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu")(x)

    for index, filters in enumerate((64, 64, 96, 96), start=1):
        # Stride 2 on alternating blocks keeps the receptive field while cutting
        # the activation working set (and therefore the TFLM tensor arena).
        stride = 2 if index in (2, 4) else 1
        x = ds_block(x, filters, f"block{index}", stride=stride)

    if architecture == "ds-cnn-temporal":
        x = temporal_head(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="wake_word_ds_cnn")


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.rows: list[dict] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val_accuracy = safe_float(logs.get("val_accuracy"), 6)
        value = float(val_accuracy or 0.0)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = None
        row = {
            "epoch": epoch + 1,
            "accuracy": safe_float(logs.get("accuracy")),
            "loss": safe_float(logs.get("loss")),
            "val_accuracy": val_accuracy,
            "val_loss": safe_float(logs.get("val_loss")),
            "learning_rate": safe_float(lr, 8),
        }
        self.rows.append(row)
        completed = epoch + 1
        progress = 0.15 + 0.67 * (completed / max(self.total_epochs, 1))
        emit_progress(
            "training",
            progress,
            f"Epoch {completed}/{self.total_epochs} · val acc {value * 100:.1f}%",
            epoch=completed,
            epochs=self.total_epochs,
            metrics=row,
        )


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (len(LABELS) * counts)
    return {index: float(weights[index]) for index in range(len(LABELS))}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix.tolist()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    true_positive = matrix[WAKE_CLASS][WAKE_CLASS]
    false_positive = matrix[0][WAKE_CLASS]
    false_negative = matrix[WAKE_CLASS][0]
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": safe_float(accuracy, 6),
        "precision": safe_float(precision, 6),
        "recall": safe_float(recall, 6),
        "f1": safe_float(f1, 6),
        "confusion_matrix": matrix,
    }


def recalibrate_batch_norm(
    model: tf.keras.Model,
    x_train: np.ndarray,
    batch_size: int,
    max_samples: int = BN_RECALIBRATION_SAMPLES,
) -> int:
    """Replace BN moving statistics with statistics measured on the training set.

    Returns the number of samples used. Training-mode forward passes use batch
    statistics, which is exactly what we want to measure; per-batch moments are
    then fused into exact dataset moments (including between-batch variance).
    """
    bn_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    if not bn_layers:
        return 0

    count = min(len(x_train), max_samples)
    if count < len(x_train):
        rng = np.random.default_rng(1337)
        indices = np.sort(rng.choice(len(x_train), size=count, replace=False))
        sample = x_train[indices]
    else:
        sample = x_train

    probe = tf.keras.Model(model.input, [layer.input for layer in bn_layers])
    moments = [
        {
            "count": 0.0,
            "sum": np.zeros(layer.moving_mean.shape, dtype=np.float64),
            "sum_sq": np.zeros(layer.moving_mean.shape, dtype=np.float64),
        }
        for layer in bn_layers
    ]

    for start in range(0, count, max(1, batch_size)):
        batch = sample[start : start + batch_size].astype(np.float32)
        if batch.shape[0] == 0:
            continue
        activations = probe(batch, training=True)
        for moment, values in zip(moments, activations):
            array = np.asarray(values, dtype=np.float64)
            axes = tuple(range(array.ndim - 1))
            elements = float(np.prod(array.shape[:-1]))
            moment["count"] += elements
            moment["sum"] += array.sum(axis=axes)
            moment["sum_sq"] += np.square(array).sum(axis=axes)

    for layer, moment in zip(bn_layers, moments):
        if moment["count"] <= 0:
            continue
        mean = moment["sum"] / moment["count"]
        variance = np.maximum(moment["sum_sq"] / moment["count"] - mean * mean, 1e-6)
        layer.moving_mean.assign(mean.astype(layer.moving_mean.dtype))
        layer.moving_variance.assign(variance.astype(layer.moving_variance.dtype))
    return count


def quantize_input(x: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    dtype = np.int8 if detail["dtype"] == np.int8 else np.uint8
    if detail["dtype"] not in (np.int8, np.uint8):
        raise ValueError(f"Unsupported TFLite input dtype: {detail['dtype']}")
    q = np.round(x / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def evaluate_tflite(tflite_model: bytes, x: np.ndarray, y: np.ndarray) -> dict:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["shape"][1] != x.shape[1] or input_detail["shape"][2] != x.shape[2]:
        raise ValueError(
            f"TFLite input shape {input_detail['shape']} does not match data {x.shape}"
        )
    predictions: list[int] = []
    for sample in x:
        q = quantize_input(sample[None, ..., None], input_detail)
        interpreter.set_tensor(input_detail["index"], q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        predictions.append(int(np.argmax(output)))
    predicted = np.asarray(predictions, dtype=np.int32)
    metrics = classification_metrics(y, predicted)
    metrics.update(
        {
            "input_shape": [int(v) for v in input_detail["shape"]],
            "input_dtype": np.dtype(input_detail["dtype"]).name,
            "input_scale": float(input_detail["quantization"][0]),
            "input_zero_point": int(input_detail["quantization"][1]),
            "output_shape": [int(v) for v in output_detail["shape"]],
            "output_dtype": np.dtype(output_detail["dtype"]).name,
            "output_scale": float(output_detail["quantization"][0]),
            "output_zero_point": int(output_detail["quantization"][1]),
        }
    )
    return metrics


def c_float(value: float, digits: int = 9) -> str:
    """Format a float as a valid C float literal (always includes a dot)."""
    text = f"{float(value):.{digits}g}"
    if not any(marker in text for marker in ".eE"):
        text += ".0"
    return text + "f"


def c_array(data: bytes, per_line: int = 16) -> str:
    lines: list[str] = []
    for start in range(0, len(data), per_line):
        chunk = data[start : start + per_line]
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines)


def write_model_header(
    path: Path,
    model_bytes: bytes,
    phrase: str,
    metrics: dict,
    tflite_metrics: dict,
    input_shape: tuple[int, ...],
    architecture: str = "DS-CNN",
) -> None:
    guard = "WAKE_WORD_MODEL_H"
    phrase_comment = phrase.replace("\n", " ").strip()
    content = f"""/* Auto-generated by WakeForge Studio.
 * Wake phrase : {phrase_comment}
 * Architecture: {architecture}
 * Input       : {list(input_shape)} int8 log-Mel
 * Test acc    : {metrics.get("accuracy")}
 * TFLite acc  : {tflite_metrics.get("accuracy")}
 *
 * The array is aligned for TFLite Micro. Keep it in flash (const).
 */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#if defined(__GNUC__)
__attribute__((aligned(16)))
#endif
const uint8_t model_data[] = {{
{c_array(model_bytes)}
}};

const unsigned int model_data_len = {len(model_bytes)}u;

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    path.write_text(content, encoding="utf-8")


def write_config_header(
    path: Path,
    phrase: str,
    metadata: dict,
    input_quant: tuple[float, int],
    output_quant: tuple[float, int],
) -> None:
    pre = metadata["preprocessing"]
    input_shape = metadata["input"]["shape"]
    guard = "WAKE_WORD_CONFIG_H"
    content = f"""/* Auto-generated by WakeForge Studio. Preprocessing must match training. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define WAKE_WORD_LABEL_COUNT {len(LABELS)}
#define WAKE_WORD_WAKE_INDEX {WAKE_CLASS}
#define WW_SAMPLE_RATE {SAMPLE_RATE}
#define WW_CLIP_SAMPLES {int(round(metadata["clip_seconds"] * SAMPLE_RATE))}
#define WW_WINDOW_SAMPLES {WIN_LENGTH}
#define WW_HOP_SAMPLES {HOP_LENGTH}
#define WW_FFT_SIZE {N_FFT}
#define WW_MEL_BINS {N_MELS}
#define WW_FRAME_COUNT {input_shape[1]}
#define WW_FMIN_HZ {c_float(FMIN, 8)}
#define WW_FMAX_HZ {c_float(FMAX, 8)}
#define WW_WINDOW_OFFSET {WINDOW_OFFSET}
#define WW_SPECTRUM_BINS {N_FFT // 2 + 1}

#define WW_INPUT_SCALE {c_float(input_quant[0], 12)}
#define WW_INPUT_ZERO_POINT {input_quant[1]}
#define WW_OUTPUT_SCALE {c_float(output_quant[0], 12)}
#define WW_OUTPUT_ZERO_POINT {output_quant[1]}

/* Normalize log-Mel features as (logmel - mean) / std before quantization. */
#define WW_FEATURE_MEAN {c_float(pre["logmel_mean"], 12)}
#define WW_FEATURE_STD {c_float(pre["logmel_std"], 12)}

/* See wake_word_mel.h for the matching window and sparse mel filterbank. */

#ifdef __cplusplus
extern "C" {{
#endif

extern const uint8_t model_data[];
extern const unsigned int model_data_len;

#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    path.write_text(content, encoding="utf-8")


def c_float_array(name: str, values: list[float], per_line: int = 8) -> str:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        lines.append("  " + ", ".join(c_float(value) for value in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def mel_filterbank_arrays() -> tuple[list[tuple[int, int, int]], list[float], list[float]]:
    """Return librosa's exact Slaney mel filters in a sparse, MCU-friendly form."""
    mel = np.asarray(
        librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            htk=False,
            norm="slaney",
        ),
        dtype=np.float64,
    )
    spans: list[tuple[int, int, int]] = []
    weights: list[float] = []
    weight_offset = 0
    for row in mel:
        active = np.flatnonzero(row > 0.0)
        if active.size == 0:
            spans.append((0, weight_offset, 0))
            continue
        bin_offset = int(active[0])
        count = int(active[-1] - active[0]) + 1
        spans.append((bin_offset, weight_offset, count))
        weights.extend(float(value) for value in row[bin_offset : bin_offset + count])
        weight_offset += count
    window = np.asarray(
        librosa.filters.get_window("hann", WIN_LENGTH, fftbins=True), dtype=np.float64
    )
    return spans, weights, [float(value) for value in window]


def write_mel_header(
    path: Path,
    phrase: str,
    spans: list[tuple[int, int, int]],
    weights: list[float],
    window: list[float],
) -> None:
    guard = "WAKE_WORD_MEL_H"
    phrase_comment = phrase.replace("\n", " ").strip()
    span_lines = "\n".join(
        f"  {{{bin_offset}, {weight_offset}, {count}}},"
        for bin_offset, weight_offset, count in spans
    )
    content = f"""/* Auto-generated by WakeForge Studio.
 * Wake phrase: {phrase_comment}
 *
 * Exact frontend constants for the log-Mel features used during training:
 *   16 kHz mono, Hann window of {WIN_LENGTH} samples centered in a {N_FFT}-point
 *   FFT, {N_MELS} Slaney-normalized mel filters, fmin={FMIN:g} Hz, fmax={FMAX:g} Hz.
 *
 * Each filter is stored as a contiguous span of FFT power bins. The weights for
 * all filters are concatenated in ww_mel_weights[]; ww_mel_spans[] points into
 * that array. This avoids a dense {N_MELS}x{N_FFT // 2 + 1} float table in flash.
 */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define WW_WINDOW_LENGTH {WIN_LENGTH}
#define WW_WINDOW_OFFSET {WINDOW_OFFSET}
#define WW_FFT_SIZE {N_FFT}
#define WW_SPECTRUM_BINS {N_FFT // 2 + 1}
#define WW_MEL_FILTERS {N_MELS}
#define WW_MEL_WEIGHT_COUNT {len(weights)}
#define WW_LOG_EPSILON 1e-6f
/* Set to N*N if your real FFT scales its forward transform by 1/N. */
#define WW_FFT_POWER_SCALE 1.0f

typedef struct {{
  uint16_t bin_offset;
  uint16_t weight_offset;
  uint16_t count;
}} ww_mel_span_t;

#ifdef __cplusplus
extern "C" {{
#endif

/* Periodic Hann window, exactly librosa.filters.get_window("hann", {WIN_LENGTH}). */
{c_float_array("ww_hann_window", window, 6)}

/* {N_MELS} filters in 0..{N_FFT // 2} FFT-power bins. */
static const ww_mel_span_t ww_mel_spans[{N_MELS}] = {{
{span_lines}
}};

{c_float_array("ww_mel_weights", weights, 8)}
#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    path.write_text(content, encoding="utf-8")


def write_preprocessing_doc(
    path: Path,
    phrase: str,
    metadata: dict,
    spans: list[tuple[int, int, int]],
) -> None:
    pre = metadata["preprocessing"]
    clip_samples = int(round(metadata["clip_seconds"] * SAMPLE_RATE))
    lines = [
        "# On-device preprocessing",
        "",
        f"Wake phrase: **{phrase}**",
        "",
        "This note describes the exact frontend used by the training pipeline. "
        "Use `wake_word_mel.h` for the window and mel filterbank coefficients.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Sample rate | {SAMPLE_RATE} Hz, mono |",
        f"| Clip window | {metadata['clip_seconds']:.2f} s ({clip_samples} samples) |",
        f"| FFT size | {N_FFT} |",
        f"| Analysis window | {WIN_LENGTH} samples ({WIN_LENGTH / SAMPLE_RATE * 1000:.0f} ms), periodic Hann |",
        f"| Window offset in FFT buffer | {WINDOW_OFFSET} zeros on each side |",
        f"| Hop | {HOP_LENGTH} samples ({HOP_LENGTH / SAMPLE_RATE * 1000:.0f} ms) |",
        f"| Frames | {metadata['input']['shape'][1]} |",
        f"| Mel filters | {N_MELS}, Slaney scale + Slaney area normalization |",
        f"| Mel range | {FMIN:g} Hz to {FMAX:g} Hz |",
        "| Log floor | `logf(mel + 1e-6f)` |",
        f"| Feature mean | `{pre['logmel_mean']:.9g}` |",
        f"| Feature std | `{pre['logmel_std']:.9g}` |",
        f"| Input scale | `{metadata['input']['quantization']['scale']:.12g}` |",
        f"| Input zero point | `{metadata['input']['quantization']['zero_point']}` |",
        f"| Wake class index | `{WAKE_CLASS}` of `{LABELS}` |",
        "",
        "## Exact steps",
        "",
        "1. Convert 16-bit PCM to float: `pcm / 32768.0f`.",
        "2. Take exactly the clip window. If it is shorter, zero-pad it (the desktop "
        "pipeline centers shorter clips; for live capture always feed a full window).",
        f"3. For frame `f` (0..{metadata['input']['shape'][1] - 1}), start at sample "
        f"`f * {HOP_LENGTH}` and form a {N_FFT}-sample buffer:",
        f"   - zero the whole buffer;",
        f"   - multiply samples `{WINDOW_OFFSET} .. {WINDOW_OFFSET + WIN_LENGTH - 1}` by "
        "`ww_hann_window[]`.",
        "4. Run an **unscaled** real FFT and keep the first 257 bins. Power is "
        "`re*re + im*im`. If your FFT scales by `1/N`, multiply by `N*N` "
        "(`WW_FFT_POWER_SCALE`).",
        "5. For each mel filter `m`, multiply the contiguous power bins by "
        "`ww_mel_weights[span.weight_offset ...]` and sum. Add `1e-6f` and take `logf`.",
        "6. Normalize: `(logmel - WW_FEATURE_MEAN) / WW_FEATURE_STD`.",
        "7. Quantize: `q = lrintf(normalized / WW_INPUT_SCALE) + WW_INPUT_ZERO_POINT`, "
        "clamp to `-128..127`, and store row-major at `input[f * WW_MEL_BINS + m]`.",
        "8. Run the TFLite Micro model. Dequantize the two outputs with "
        "`WW_OUTPUT_SCALE` and `WW_OUTPUT_ZERO_POINT`; class 1 is the wake word.",
        "",
        "The mel table is sparse: each `ww_mel_span_t` gives the first FFT bin, the "
        "offset into the concatenated weight array, and the number of bins.",
        "",
        "## C sketch",
        "",
        "```c",
        "#include <math.h>",
        "#include <stdint.h>",
        '#include "wake_word_config.h"',
        '#include "wake_word_mel.h"',
        "",
        "// fft_power() must fill 257 unscaled power bins. Add FFT scaling if needed.",
        "void wake_frontend(const int16_t *pcm, int8_t *input) {",
        f"  static float frame[WW_FFT_SIZE];",
        f"  static float power[WW_SPECTRUM_BINS];",
        "  for (int f = 0; f < WW_FRAME_COUNT; ++f) {",
        "    for (int k = 0; k < WW_FFT_SIZE; ++k) frame[k] = 0.0f;",
        "    const int base = f * WW_HOP_SAMPLES;",
        "    for (int n = 0; n < WW_WINDOW_SAMPLES; ++n) {",
        "      frame[WW_WINDOW_OFFSET + n] =",
        "          (pcm[base + WW_WINDOW_OFFSET + n] / 32768.0f) * ww_hann_window[n];",
        "    }",
        "    fft_power(frame, power);",
        "    for (int m = 0; m < WW_MEL_FILTERS; ++m) {",
        "      const ww_mel_span_t *span = &ww_mel_spans[m];",
        "      float acc = 0.0f;",
        "      for (int j = 0; j < span->count; ++j) {",
        "        acc += power[span->bin_offset + j] *",
        "               ww_mel_weights[span->weight_offset + j];",
        "      }",
        "      const float logmel = logf(acc + WW_LOG_EPSILON);",
        "      const float normalized = (logmel - WW_FEATURE_MEAN) / WW_FEATURE_STD;",
        "      long q = lrintf(normalized / WW_INPUT_SCALE) + WW_INPUT_ZERO_POINT;",
        "      if (q < -128) q = -128;",
        "      if (q > 127) q = 127;",
        "      input[f * WW_MEL_BINS + m] = (int8_t)q;",
        "    }",
        "  }",
        "}",
        "```",
        "",
        "For a live detector, run this on a sliding full window and require a few "
        "consecutive high wake scores before emitting an event. A simple threshold "
        "on the dequantized class-1 score is usually enough to start tuning.",
        "",
        f"The filter table contains {len(spans)} spans and "
        f"{sum(span[2] for span in spans)} weights "
        "(about a few KiB in flash instead of a dense 40x257 float matrix).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def history_payload(
    callback: ProgressCallback,
    early_stopping: tf.keras.callbacks.EarlyStopping,
    requested_epochs: int,
) -> dict:
    """Describe the epoch whose weights the model actually ended up with."""
    rows = callback.rows
    if early_stopping.best_weights is not None:
        # EarlyStopping restored these weights in on_train_end.
        best_epoch = int(early_stopping.best_epoch) + 1
        restored = True
    else:
        # Training ended before start_from_epoch: the final weights are kept.
        best_epoch = len(rows)
        restored = False
    best_row = rows[best_epoch - 1] if 0 < best_epoch <= len(rows) else {}
    return {
        "epochs": rows,
        "epochs_requested": requested_epochs,
        "epochs_completed": len(rows),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_row.get("val_accuracy"),
        "best_val_loss": best_row.get("val_loss"),
        "selection_metric": EARLY_STOP_MONITOR,
        "weights_restored": restored,
    }


def write_metadata(
    path: Path,
    phrase: str,
    args: argparse.Namespace,
    metrics: dict,
    tflite_metrics: dict,
    history: dict,
    counts: dict,
    model_params: int,
    tflite_size: int,
    input_quant: tuple[float, int],
    output_quant: tuple[float, int],
    feature_mean: float,
    feature_std: float,
    frame_count: int,
    bn_samples: int,
    files: dict,
) -> dict:
    metadata = {
        "phrase": phrase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURES[args.architecture],
        "labels": LABELS,
        "clip_seconds": args.clip_seconds,
        "sample_rate": SAMPLE_RATE,
        "input": {
            "shape": [1, frame_count, N_MELS, 1],
            "dtype": tflite_metrics["input_dtype"],
            "quantization": {
                "scale": tflite_metrics["input_scale"],
                "zero_point": tflite_metrics["input_zero_point"],
            },
        },
        "output": {
            "shape": tflite_metrics["output_shape"],
            "dtype": tflite_metrics["output_dtype"],
            "quantization": {
                "scale": tflite_metrics["output_scale"],
                "zero_point": tflite_metrics["output_zero_point"],
            },
        },
        "preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "window_samples": WIN_LENGTH,
            "hop_samples": HOP_LENGTH,
            "fft_size": N_FFT,
            "mel_bins": N_MELS,
            "fmin_hz": FMIN,
            "fmax_hz": FMAX,
            "frame_count": frame_count,
            "logmel_mean": feature_mean,
            "logmel_std": feature_std,
        },
        "metrics": {
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "tflite_test_accuracy": tflite_metrics["accuracy"],
            "tflite_test_precision": tflite_metrics["precision"],
            "tflite_test_recall": tflite_metrics["recall"],
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": history["epochs_completed"],
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "best_epoch": history["best_epoch"],
            "best_val_accuracy": history["best_val_accuracy"],
            "best_val_loss": history["best_val_loss"],
            "selection_metric": history["selection_metric"],
            "dataset_counts": counts,
            "bn_momentum": BN_MOMENTUM,
            "bn_recalibration_samples": bn_samples,
        },
        "model": {
            "parameters": int(model_params),
            "tflite_size_bytes": int(tflite_size),
            "recommended_tensor_arena_bytes": 128 * 1024,
            "tensor_arena_note": (
                "Starting estimate for TFLite Micro. Measure arena_used_bytes() "
                "on the target and adjust."
            ),
        },
        "deployment": {
            "labels": LABELS,
            "wake_index": WAKE_CLASS,
            "mel_header": files.get("mel_header"),
            "preprocessing_doc": files.get("preprocessing_doc"),
        },
        "artifacts": files,
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def staging_dir_for(output: Path) -> Path:
    return output.with_name(f".{output.name}.staging")


def publish_artifacts(staging: Path, output: Path) -> None:
    """Replace ``output`` with the fully written ``staging`` folder."""
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
    except OSError:
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset).expanduser().resolve()
    final_output = Path(args.output).expanduser().resolve()
    # Every artifact is written to a staging folder first and swapped in only
    # after the whole export succeeded, so a failed or canceled run can never
    # leave a new .tflite next to headers with the previous quantization.
    output = staging_dir_for(final_output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise ValueError(f"dataset_info.json not found in {dataset_dir}. Build the dataset first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    phrase = args.phrase.strip() or info.get("phrase", "wake word")
    clip_seconds = float(info.get("clip_seconds", args.clip_seconds))
    if abs(clip_seconds - args.clip_seconds) > 1e-6:
        log(f"Using clip length from dataset: {clip_seconds}s")
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))
    expected_frames = 1 + (clip_samples - N_FFT) // HOP_LENGTH
    log(f"Phrase: {phrase!r} · clip {clip_seconds:.2f}s · {expected_frames} frames")

    emit_progress("load", 0.02, "Loading recordings")
    x_train, y_train = load_split(dataset_dir, "train", clip_samples)
    x_val, y_val = load_split(dataset_dir, "val", clip_samples)
    x_test, y_test = load_split(dataset_dir, "test", clip_samples)
    counts = {
        split: {
            label: int(np.sum(labels == index))
            for index, label in enumerate(LABELS)
        }
        for split, labels in (("train", y_train), ("val", y_val), ("test", y_test))
    }
    log(f"Dataset counts: {json.dumps(counts)}")

    emit_progress("prepare", 0.08, "Normalizing log-Mel features")
    feature_mean = float(np.mean(x_train))
    feature_std = float(np.std(x_train) + 1e-5)
    x_train = ((x_train - feature_mean) / feature_std).astype(np.float32)
    x_val = ((x_val - feature_mean) / feature_std).astype(np.float32)
    x_test = ((x_test - feature_mean) / feature_std).astype(np.float32)
    log(f"Feature normalization: mean={feature_mean:.5f} std={feature_std:.5f}")

    train_ds = make_tf_dataset(x_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_tf_dataset(x_val, y_val, args.batch_size, shuffle=False)
    test_ds = make_tf_dataset(x_test, y_test, args.batch_size, shuffle=False)

    emit_progress("build", 0.12, f"Building {ARCHITECTURES[args.architecture]}")
    model = build_model((expected_frames, N_MELS, 1), args.architecture)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log)

    class_weight = balanced_class_weights(y_train)
    log(f"Class weights: {class_weight}")
    progress_callback = ProgressCallback(args.epochs)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=EARLY_STOP_MONITOR,
        mode="min",
        patience=args.patience,
        start_from_epoch=min(MIN_EPOCHS_BEFORE_EARLY_STOP, args.epochs),
        restore_best_weights=True,
        verbose=0,
    )
    callbacks = [
        progress_callback,
        early_stopping,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    emit_progress("training", 0.15, f"Training for up to {args.epochs} epochs")
    start_epoch = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=False,
        verbose=0,
    )
    log(f"Training finished in {time.time() - start_epoch:.1f}s")
    history = history_payload(progress_callback, early_stopping, args.epochs)
    if history["weights_restored"]:
        log(
            f"Using weights from epoch {history['best_epoch']} "
            f"(lowest {EARLY_STOP_MONITOR} after epoch {MIN_EPOCHS_BEFORE_EARLY_STOP})"
        )

    emit_progress("calibrate", 0.83, "Recalibrating BatchNorm statistics")
    bn_samples = recalibrate_batch_norm(model, x_train, args.batch_size)
    log(f"BatchNorm recalibrated on {bn_samples} training samples")

    emit_progress("evaluate", 0.84, "Evaluating float model")
    float_loss, float_accuracy = model.evaluate(test_ds, verbose=0)
    float_predictions = np.argmax(model.predict(x_test[..., None], verbose=0), axis=1)
    metrics = classification_metrics(y_test, float_predictions)
    metrics["loss"] = safe_float(float_loss)
    metrics["float_accuracy"] = safe_float(float_accuracy)
    confusion = metrics["confusion_matrix"]
    log(f"Float test accuracy: {metrics['accuracy']:.4f}")
    log(f"Confusion matrix (rows=true background/wake): {confusion}")

    emit_progress("convert", 0.88, "Converting to fully-int8 TFLite")

    def representative_dataset():
        # Deterministic, class-balanced subset keeps conversion fast and stable.
        max_samples = min(len(x_train), 400)
        indices = np.linspace(0, len(x_train) - 1, num=max_samples, dtype=np.int64)
        for index in indices:
            yield [x_train[index][None, ..., None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    tflite_path = output / "wake_word_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    log(f"TFLite model: {len(tflite_model) / 1024:.1f} KiB")

    emit_progress("verify", 0.93, "Verifying quantized accuracy")
    tflite_metrics = evaluate_tflite(tflite_model, x_test, y_test)
    log(f"INT8 test accuracy: {tflite_metrics['accuracy']:.4f}")
    log(f"INT8 confusion matrix: {tflite_metrics['confusion_matrix']}")

    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model.save(output / "wake_word.keras")

    header_path = output / "wake_word_model.h"
    config_path = output / "wake_word_config.h"
    mel_path = output / "wake_word_mel.h"
    preprocessing_doc_path = output / "PREPROCESSING.md"
    metadata_path = output / "metadata.json"
    files = {
        "keras": "wake_word.keras",
        "tflite": "wake_word_int8.tflite",
        "header": "wake_word_model.h",
        "config_header": "wake_word_config.h",
        "mel_header": "wake_word_mel.h",
        "preprocessing_doc": "PREPROCESSING.md",
        "metadata": "metadata.json",
        "history": "history.json",
    }
    emit_progress("export", 0.96, "Writing C headers and metadata")
    input_quant = (tflite_metrics["input_scale"], tflite_metrics["input_zero_point"])
    output_quant = (tflite_metrics["output_scale"], tflite_metrics["output_zero_point"])
    mel_spans, mel_weights, hann_window = mel_filterbank_arrays()
    write_model_header(
        header_path,
        tflite_model,
        phrase,
        metrics,
        tflite_metrics,
        tuple(tflite_metrics["input_shape"]),
        ARCHITECTURES[args.architecture],
    )
    metadata = write_metadata(
        metadata_path,
        phrase,
        argparse.Namespace(**{**vars(args), "clip_seconds": clip_seconds}),
        metrics,
        tflite_metrics,
        history,
        counts,
        model.count_params(),
        len(tflite_model),
        input_quant,
        output_quant,
        feature_mean,
        feature_std,
        expected_frames,
        bn_samples,
        files,
    )
    write_config_header(config_path, phrase, metadata, input_quant, output_quant)
    write_mel_header(mel_path, phrase, mel_spans, mel_weights, hann_window)
    write_preprocessing_doc(preprocessing_doc_path, phrase, metadata, mel_spans)

    # A compact manifest for firmware/CI consumers.
    export_info = {
        "phrase": phrase,
        "created_at": metadata["created_at"],
        "labels": LABELS,
        "input_shape": metadata["input"]["shape"],
        "files": files,
        "test_accuracy": metrics["accuracy"],
        "tflite_test_accuracy": tflite_metrics["accuracy"],
    }
    (output / "export_info.json").write_text(
        json.dumps(export_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    publish_artifacts(output, final_output)
    output = final_output

    duration = time.time() - started
    log(f"Done in {duration:.1f}s. Artifacts in {output}")
    emit_progress("done", 1.0, "Training and export complete")
    emit_result(
        {
            "output": str(output),
            "phrase": phrase,
            "metrics": {
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "tflite_test_accuracy": tflite_metrics["accuracy"],
                "test_loss": metrics["loss"],
            },
            "model_size_bytes": len(tflite_model),
            "epochs_completed": history["epochs_completed"],
            "files": files,
            "duration_seconds": round(duration, 2),
        }
    )
    return {
        "output": str(output),
        "metrics": metrics,
        "tflite_metrics": tflite_metrics,
        "files": files,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and quantize a wake-word DS-CNN")
    parser.add_argument("--dataset", required=True, help="Augmented dataset folder")
    parser.add_argument("--output", required=True, help="Output folder for model artifacts")
    parser.add_argument("--phrase", default="wake word", help="Wake phrase used for metadata")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clip-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="ds-cnn-temporal")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.epochs > 1000:
        parser.error("--epochs must be between 1 and 1000")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def main() -> int:
    configure_stdio()
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        # The previous model in --output is untouched; drop the partial run.
        staging = staging_dir_for(Path(args.output).expanduser().resolve())
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[train] ERROR: {exc}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
