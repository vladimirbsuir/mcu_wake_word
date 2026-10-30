//! Microphone capture, live metering and voice-activity trimming.
//!
//! cpal gives us the device's native sample rate/format. We downmix to mono in
//! the callback, then resample to 16 kHz with a band-limited (windowed-sinc)
//! filter and trim leading/trailing silence when the user presses Stop. This avoids requiring the OS to expose a 16 kHz input
//! mode (many Windows/WASAPI endpoints do not).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use serde::Serialize;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
const MAX_CAPTURE_SECONDS: usize = 30;
const FRAME_MS: usize = 20;
const FRAME_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * FRAME_MS / 1000;
const PAD_FRAMES: usize = 8; // 160 ms of pre/post roll
const MIN_SPEECH_MS: usize = 250;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStart {
    pub id: String,
    pub device_name: String,
    pub input_sample_rate: u32,
    pub channels: u16,
    pub max_seconds: u32,
    pub started_at: String,
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Meter {
    pub levels: Vec<f32>,
    pub rms: f32,
    pub peak: f32,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrimReport {
    pub threshold: f32,
    pub original_ms: u64,
    pub speech_ms: u64,
    pub trimmed_ms: u64,
    pub trim_start_ms: u64,
    pub trim_end_ms: u64,
    pub resampled_from: u32,
}

#[derive(Debug)]
pub struct ProcessedCapture {
    pub samples: Vec<f32>,
    pub report: TrimReport,
}

#[derive(Debug)]
pub struct CaptureResult {
    pub samples: Vec<f32>,
    pub input_sample_rate: u32,
    pub device_name: String,
    pub elapsed_ms: u64,
}

struct ReadyInfo {
    device_name: String,
    sample_rate: u32,
}

pub struct CaptureSession {
    pub id: String,
    pub started_at: Instant,
    pub started_at_rfc3339: String,
    thread: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: Arc<AtomicU32>,
    channels: Arc<AtomicU16Wrapper>,
    error: Arc<Mutex<Option<String>>>,
    device_name: String,
    input_sample_rate: u32,
}

// AtomicU16 exists, but keeping this tiny wrapper makes ownership explicit.
#[derive(Debug)]
struct AtomicU16Wrapper(AtomicU32);

impl AtomicU16Wrapper {
    fn new(value: u16) -> Self {
        Self(AtomicU32::new(u32::from(value)))
    }
    fn load(&self) -> u16 {
        self.0.load(Ordering::Relaxed) as u16
    }
}

impl CaptureSession {
    pub fn start(id: String) -> Result<Self, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let sample_rate = Arc::new(AtomicU32::new(0));
        let channels = Arc::new(AtomicU16Wrapper::new(0));
        let error = Arc::new(Mutex::new(None::<String>));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<ReadyInfo, String>>();

        let thread_stop = Arc::clone(&stop);
        let thread_samples = Arc::clone(&samples);
        let thread_rate = Arc::clone(&sample_rate);
        let thread_channels = Arc::clone(&channels);
        let thread_error = Arc::clone(&error);
        let started_at = Instant::now();
        let started_at_rfc3339 = chrono::Utc::now().to_rfc3339();

        let thread = thread::Builder::new()
            .name("wakeforge-audio-capture".into())
            .spawn(move || {
                let setup = (|| -> Result<(cpal::Stream, ReadyInfo, String), String> {
                    let host = cpal::default_host();
                    let device = host
                        .default_input_device()
                        .ok_or_else(|| "No microphone found. Connect an input device and try again.".to_string())?;
                    let device_name = device.name().unwrap_or_else(|_| "Default microphone".into());
                    let supported = device
                        .default_input_config()
                        .map_err(|error| format!("Cannot query microphone format: {error}"))?;
                    let sample_format = supported.sample_format();
                    let config: StreamConfig = supported.into();
                    let rate = config.sample_rate.0;
                    let channel_count = config.channels;
                    if channel_count == 0 {
                        return Err("Microphone reports zero channels.".into());
                    }
                    thread_rate.store(rate, Ordering::SeqCst);
                    thread_channels.0.store(u32::from(channel_count), Ordering::SeqCst);

                    let max_samples = rate as usize * MAX_CAPTURE_SECONDS;
                    let error_for_callback = Arc::clone(&thread_error);
                    let stream = match sample_format {
                        SampleFormat::F32 => build_stream::<f32>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::F64 => build_stream::<f64>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::I8 => build_stream::<i8>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::I16 => build_stream::<i16>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::I32 => build_stream::<i32>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::I64 => build_stream::<i64>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::U8 => build_stream::<u8>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::U16 => build_stream::<u16>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::U32 => build_stream::<u32>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        SampleFormat::U64 => build_stream::<u64>(
                            &device,
                            &config,
                            channel_count,
                            Arc::clone(&thread_samples),
                            max_samples,
                            Arc::clone(&error_for_callback),
                        ),
                        other => Err(format!("Unsupported microphone sample format: {other:?}")),
                    }?;

                    stream
                        .play()
                        .map_err(|error| format!("Cannot start microphone stream: {error}"))?;
                    Ok((
                        stream,
                        ReadyInfo {
                            device_name: device_name.clone(),
                            sample_rate: rate,
                        },
                        device_name,
                    ))
                })();

                let (stream, ready, device_name) = match setup {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };

                if ready_tx.send(Ok(ready)).is_err() {
                    return;
                }

                // Keep the stream alive on this thread. The callback owns a
                // clone of the sample buffer and runs on the audio thread.
                while !thread_stop.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(25));
                }
                drop(stream);
                let _ = device_name;
            })
            .map_err(|error| format!("Cannot spawn audio capture thread: {error}"))?;

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(ready)) => Ok(Self {
                id,
                started_at,
                started_at_rfc3339,
                thread: Some(thread),
                stop,
                samples,
                sample_rate,
                channels,
                error,
                device_name: ready.device_name,
                input_sample_rate: ready.sample_rate,
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                stop.store(true, Ordering::SeqCst);
                let _ = thread.join();
                Err("Timed out while opening the microphone.".into())
            }
        }
    }

    pub fn start_info(&self) -> CaptureStart {
        CaptureStart {
            id: self.id.clone(),
            device_name: self.device_name.clone(),
            input_sample_rate: self.input_sample_rate,
            channels: self.channels.load(),
            max_seconds: MAX_CAPTURE_SECONDS as u32,
            started_at: self.started_at_rfc3339.clone(),
        }
    }

    pub fn meter(&self) -> Meter {
        let samples = match self.samples.lock() {
            Ok(buffer) => buffer,
            Err(_) => return Meter::default(),
        };
        if samples.is_empty() {
            return Meter::default();
        }
        let rate = self.sample_rate.load(Ordering::Relaxed).max(1) as f64;
        let total_frames = samples.len();
        let window = if total_frames > (rate as usize * 3) {
            &samples[total_frames - rate as usize * 3..]
        } else {
            &samples[..]
        };
        let peak = window.iter().fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
        let rms = (window
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / window.len().max(1) as f64)
            .sqrt() as f32;

        let bars = 96usize;
        let chunk = (window.len() / bars).max(1);
        let mut levels = Vec::with_capacity(bars);
        let visual_peak = peak.max(0.012);
        for chunk_start in (0..window.len()).step_by(chunk) {
            let chunk_end = (chunk_start + chunk).min(window.len());
            let local_peak = window[chunk_start..chunk_end]
                .iter()
                .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
            levels.push((local_peak / visual_peak).powf(0.62).clamp(0.0, 1.0));
        }
        if levels.len() > bars {
            // The final partial chunk can produce one extra bar.
            levels.truncate(bars);
        }

        Meter {
            levels,
            rms,
            peak,
            duration_ms: (total_frames as f64 / rate * 1000.0).round() as u64,
        }
    }

    pub fn stop(mut self) -> Result<CaptureResult, String> {
        let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| "Microphone capture thread crashed.".to_string())?;
        }
        if let Ok(error) = self.error.lock() {
            if let Some(message) = error.as_ref() {
                return Err(format!("Microphone stream error: {message}"));
            }
        }
        let samples = self
            .samples
            .lock()
            .map_err(|_| "Microphone buffer is unavailable.".to_string())?
            .clone();
        if samples.is_empty() {
            return Err("No audio was captured. Check the microphone and try again.".into());
        }
        Ok(CaptureResult {
            samples,
            input_sample_rate: self.input_sample_rate,
            device_name: self.device_name,
            elapsed_ms,
        })
    }

    pub fn cancel(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: u16,
    samples: Arc<Mutex<Vec<f32>>>,
    max_samples: usize,
    error_slot: Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + Sample + 'static,
    f32: FromSample<T>,
{
    let channel_count = usize::from(channels.max(1));
    let error_callback = Arc::clone(&error_slot);
    device
        .build_input_stream(
            config,
            move |data: &[T], _info: &cpal::InputCallbackInfo| {
                if let Ok(mut buffer) = samples.lock() {
                    if buffer.len() >= max_samples {
                        return;
                    }
                    for frame in data.chunks(channel_count) {
                        if buffer.len() >= max_samples {
                            break;
                        }
                        let mut sum = 0.0_f32;
                        for sample in frame {
                            sum += sample.to_sample::<f32>();
                        }
                        buffer.push(sum / channel_count as f32);
                    }
                }
            },
            move |error| {
                if let Ok(mut slot) = error_callback.lock() {
                    if slot.is_none() {
                        *slot = Some(error.to_string());
                    }
                }
            },
            None,
        )
        .map_err(|error| error.to_string())
}

// Windowed-sinc resampler parameters. The cutoff sits at 0.47 of the lower
// rate (7.52 kHz for 16 kHz output) and the Kaiser window gives ~60 dB of
// stopband by the output Nyquist, so fricative energy above 8 kHz does not
// fold back into the 0-7.6 kHz mel range the model is trained on.
const RESAMPLE_CUTOFF: f64 = 0.47;
const RESAMPLE_ZERO_CROSSINGS: f64 = 32.0;
const RESAMPLE_KAISER_BETA: f64 = 5.65;
const RESAMPLE_TABLE_STEPS: usize = 512; // kernel table entries per input sample

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..64 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}

/// Deterministic, dependency-free band-limited resampler (Kaiser-windowed
/// sinc). The kernel is tabulated once and linearly interpolated per tap.
fn resample(input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if source_rate == 0 || target_rate == 0 {
        return Err("Microphone reported an invalid sample rate.".into());
    }
    if source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let ratio = f64::from(source_rate) / f64::from(target_rate);
    // All distances below are measured in input samples.
    let scale = (1.0 / ratio).min(1.0);
    let cutoff = RESAMPLE_CUTOFF * scale; // cycles per input sample
    let half_width = RESAMPLE_ZERO_CROSSINGS / (2.0 * cutoff);

    let table_len = (half_width * RESAMPLE_TABLE_STEPS as f64).ceil() as usize + 2;
    let i0_beta = bessel_i0(RESAMPLE_KAISER_BETA);
    let table: Vec<f64> = (0..table_len)
        .map(|step| {
            let x = step as f64 / RESAMPLE_TABLE_STEPS as f64;
            if x >= half_width {
                return 0.0;
            }
            let arg = 2.0 * cutoff * x;
            let sinc = if arg == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
            };
            let r = x / half_width;
            let window = bessel_i0(RESAMPLE_KAISER_BETA * (1.0 - r * r).sqrt()) / i0_beta;
            2.0 * cutoff * sinc * window
        })
        .collect();
    let kernel = |distance: f64| -> f64 {
        let position = distance * RESAMPLE_TABLE_STEPS as f64;
        let index = position as usize;
        if index + 1 >= table.len() {
            return 0.0;
        }
        let fraction = position - index as f64;
        table[index] + (table[index + 1] - table[index]) * fraction
    };

    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let last = input.len() as i64 - 1;
    let mut output = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let center = index as f64 * ratio;
        let first = ((center - half_width).ceil() as i64).max(0);
        let end = ((center + half_width).floor() as i64).min(last);
        let mut acc = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for tap in first..=end {
            let weight = kernel((tap as f64 - center).abs());
            acc += f64::from(input[tap as usize]) * weight;
            weight_sum += weight;
        }
        // Normalizing per output keeps unity DC gain, including at the edges.
        output.push(if weight_sum > 1e-9 {
            (acc / weight_sum) as f32
        } else {
            0.0
        });
    }
    Ok(output)
}

fn percentile(values: &mut [f32], fraction: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index]
}

pub fn process_capture(capture: &CaptureResult) -> Result<ProcessedCapture, String> {
    let original_ms = ((capture.samples.len() as f64 / capture.input_sample_rate.max(1) as f64)
        * 1000.0)
        .round() as u64;
    let mut samples = resample(&capture.samples, capture.input_sample_rate, TARGET_SAMPLE_RATE)?;
    if samples.len() < FRAME_SAMPLES * 2 {
        return Err("Recording is too short. Hold the phrase for at least half a second.".into());
    }

    let mean = samples.iter().map(|s| f64::from(*s)).sum::<f64>() / samples.len() as f64;
    for sample in &mut samples {
        *sample = (*sample - mean as f32).clamp(-1.0, 1.0);
    }

    let peak = samples.iter().fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    if peak < 0.004 {
        return Err(
            "The microphone signal is too quiet. Move closer or raise the input level.".into(),
        );
    }

    let frame_count = samples.len() / FRAME_SAMPLES;
    let mut frame_rms = Vec::with_capacity(frame_count);
    for frame in samples.chunks_exact(FRAME_SAMPLES).take(frame_count) {
        let energy = frame
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / FRAME_SAMPLES as f64;
        frame_rms.push(energy.sqrt() as f32);
    }

    let mut sorted_rms = frame_rms.clone();
    let noise_floor = percentile(&mut sorted_rms, 0.2);
    let threshold = (noise_floor * 2.8)
        .max(peak * 0.035)
        .max(0.0035)
        .min(peak * 0.5);
    let active_frames: Vec<usize> = frame_rms
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= threshold).then_some(index))
        .collect();
    if active_frames.len() < 3 {
        return Err(
            "No speech detected above the noise floor. Try again in a quieter room.".into(),
        );
    }

    let first = active_frames[0].saturating_sub(PAD_FRAMES);
    let last = (active_frames[active_frames.len() - 1] + PAD_FRAMES + 1).min(frame_count);
    let start_sample = first * FRAME_SAMPLES;
    let end_sample = (last * FRAME_SAMPLES).min(samples.len());
    let mut trimmed = samples[start_sample..end_sample].to_vec();
    let speech_ms = (trimmed.len() as f64 / TARGET_SAMPLE_RATE as f64 * 1000.0).round() as u64;
    if speech_ms < MIN_SPEECH_MS as u64 {
        return Err("Detected speech is too short. Record the complete wake phrase.".into());
    }

    // Short fades at both boundaries prevent clicks in playback and training.
    let fade = (TARGET_SAMPLE_RATE as usize / 200).min(trimmed.len() / 2); // 5 ms
    let trimmed_len = trimmed.len();
    for index in 0..fade {
        let gain = index as f32 / fade as f32;
        trimmed[index] *= gain;
        trimmed[trimmed_len - 1 - index] *= gain;
    }

    let trim_start_ms = (start_sample as f64 / TARGET_SAMPLE_RATE as f64 * 1000.0).round() as u64;
    let trim_end_ms = original_ms.saturating_sub(trim_start_ms + speech_ms);
    Ok(ProcessedCapture {
        samples: trimmed,
        report: TrimReport {
            threshold,
            original_ms,
            speech_ms,
            trimmed_ms: original_ms.saturating_sub(speech_ms),
            trim_start_ms,
            trim_end_ms,
            resampled_from: capture.input_sample_rate,
        },
    })
}

#[allow(dead_code)]
pub fn target_rate() -> u32 {
    TARGET_SAMPLE_RATE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, frequency: f64, seconds: f64) -> Vec<f32> {
        let count = (f64::from(rate) * seconds) as usize;
        (0..count)
            .map(|n| (2.0 * std::f64::consts::PI * frequency * n as f64 / f64::from(rate)).sin() as f32 * 0.5)
            .collect()
    }

    /// RMS of the middle half, away from edge effects, in dB relative to the input sine.
    fn level_db(samples: &[f32]) -> f64 {
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let rms = (middle.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / middle.len() as f64).sqrt();
        20.0 * (rms / (0.5 / 2.0_f64.sqrt())).log10()
    }

    #[test]
    fn downsampling_keeps_passband() {
        for rate in [44_100, 48_000] {
            for frequency in [300.0, 1_000.0, 4_000.0, 7_000.0] {
                let out = resample(&sine(rate, frequency, 1.0), rate, TARGET_SAMPLE_RATE).unwrap();
                let level = level_db(&out);
                assert!(level.abs() < 0.5, "{rate} Hz, {frequency} Hz tone: {level:.2} dB");
            }
        }
    }

    #[test]
    fn downsampling_rejects_aliases() {
        for rate in [44_100, 48_000] {
            for frequency in [8_500.0, 10_000.0, 14_000.0, 20_000.0] {
                let out = resample(&sine(rate, frequency, 1.0), rate, TARGET_SAMPLE_RATE).unwrap();
                let level = level_db(&out);
                assert!(level < -50.0, "{rate} Hz, {frequency} Hz tone aliased at {level:.1} dB");
            }
        }
    }

    #[test]
    fn upsampling_keeps_passband() {
        let out = resample(&sine(8_000, 1_000.0, 1.0), 8_000, TARGET_SAMPLE_RATE).unwrap();
        assert_eq!(out.len(), 16_000);
        assert!(level_db(&out).abs() < 0.5);
    }
}
