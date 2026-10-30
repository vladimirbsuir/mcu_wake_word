//! Project layout and filesystem helpers.
//!
//! A "project" is a wake phrase. Everything produced for that phrase lives in
//! `<app_data>/projects/<slug>/`:
//!
//!   recordings/   clean, VAD-trimmed takes
//!   dataset/      augmented train/val/test data
//!   models/       Keras, TFLite, C headers, metrics
//!   exports/      optional local copies
//!
//! Keeping projects separate prevents accidentally mixing samples from
//! different wake words.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use tauri::{AppHandle, Manager};

const MAX_PHRASE_CHARS: usize = 64;
const MIN_PHRASE_CHARS: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingMeta {
    pub id: String,
    pub file_name: String,
    pub duration_ms: u64,
    pub size_bytes: u64,
    pub created_at: String,
    pub phrase: String,
    pub source_sample_rate: u32,
    pub trimmed_ms: u64,
    pub vad_threshold: f32,
    /// The text shown while recording "other speech" takes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SplitCounts {
    pub wake_word: usize,
    pub background: usize,
}

impl SplitCounts {
    pub fn total(&self) -> usize {
        self.wake_word + self.background
    }
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSummary {
    pub exists: bool,
    pub train: SplitCounts,
    pub val: SplitCounts,
    pub test: SplitCounts,
    pub total: usize,
    pub clip_seconds: Option<f64>,
    pub variations_per_sample: Option<u64>,
    pub created_at: Option<String>,
    pub background_source: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub exists: bool,
    pub metadata: Option<Value>,
    pub history: Option<Value>,
    pub tflite_size_bytes: u64,
    pub header_available: bool,
    pub updated_at: Option<String>,
    /// Streaming benchmark written by evaluate.py, if it was run for this model.
    pub evaluation: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectPaths {
    pub project_dir: String,
    pub recordings_dir: String,
    pub negatives_dir: String,
    pub dataset_dir: String,
    pub models_dir: String,
    pub exports_dir: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub phrase: String,
    pub slug: String,
    pub recordings: Vec<RecordingMeta>,
    /// The speaker's own speech that must not trigger (hard negatives).
    pub negative_takes: Vec<RecordingMeta>,
    /// Windows of corpus speech an earlier model fired on (mine.py).
    pub mined_count: usize,
    pub dataset: DatasetSummary,
    pub model: ModelSummary,
    pub paths: ProjectPaths,
    /// Spoken phrase, language and word lists (see voice.rs).
    pub voice: Value,
    /// Summary of synthesized TTS clips, if any.
    pub synthetic: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectListItem {
    pub phrase: String,
    pub slug: String,
    pub recording_count: usize,
    pub has_dataset: bool,
    pub has_model: bool,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

/// Live validation result for the create/rename dialog.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectNamePreview {
    pub phrase: String,
    pub slug: String,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub destination: String,
    pub files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct RecordingSidecar {
    #[serde(default)]
    phrase: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    duration_ms: u64,
    #[serde(default)]
    source_sample_rate: u32,
    #[serde(default)]
    trimmed_ms: u64,
    #[serde(default)]
    vad_threshold: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectFile {
    pub phrase: String,
    pub slug: String,
    pub created_at: String,
    pub updated_at: String,
}

pub fn normalize_phrase(phrase: &str) -> Result<String, String> {
    let normalized = phrase.split_whitespace().collect::<Vec<_>>().join(" ");
    let char_count = normalized.chars().count();
    if char_count < MIN_PHRASE_CHARS {
        return Err("Wake phrase must contain at least 2 characters.".into());
    }
    if char_count > MAX_PHRASE_CHARS {
        return Err(format!(
            "Wake phrase is too long ({char_count} chars, max {MAX_PHRASE_CHARS})."
        ));
    }
    if !normalized.chars().any(|c| c.is_alphanumeric()) {
        return Err("Wake phrase must contain at least one letter or digit.".into());
    }
    Ok(normalized)
}

fn transliterate(ch: char) -> Option<&'static str> {
    let value = match ch {
        'а' => "a",
        'б' => "b",
        'в' => "v",
        'г' => "g",
        'д' => "d",
        'е' => "e",
        'ё' => "yo",
        'ж' => "zh",
        'з' => "z",
        'и' => "i",
        'й' => "y",
        'к' => "k",
        'л' => "l",
        'м' => "m",
        'н' => "n",
        'о' => "o",
        'п' => "p",
        'р' => "r",
        'с' => "s",
        'т' => "t",
        'у' => "u",
        'ф' => "f",
        'х' => "h",
        'ц' => "ts",
        'ч' => "ch",
        'ш' => "sh",
        'щ' => "sch",
        'ъ' => "",
        'ы' => "y",
        'ь' => "",
        'э' => "e",
        'ю' => "yu",
        'я' => "ya",
        'і' => "i",
        'ї' => "yi",
        'є' => "ye",
        'ґ' => "g",
        _ => return None,
    };
    Some(value)
}

fn stable_hash(input: &str) -> u32 {
    // FNV-1a; only used for otherwise unrepresentable project names.
    let mut hash: u32 = 2_166_136_261;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

const MAX_SLUG_CHARS: usize = 48;
const PROJECT_FILE: &str = "project.json";
const PROJECT_SUBDIRS: [&str; 5] = ["recordings", "negatives", "dataset", "models", "exports"];

/// Where a take is stored: the wake phrase itself, or other speech of the
/// same speaker that must not wake the device (used as hard negatives).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeKind {
    Wake,
    Negative,
}

impl TakeKind {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("wake") {
            "wake" => Ok(Self::Wake),
            "negative" => Ok(Self::Negative),
            other => Err(format!("Unknown take kind '{other}'.")),
        }
    }

    pub fn folder(self) -> &'static str {
        match self {
            Self::Wake => "recordings",
            Self::Negative => "negatives",
        }
    }
}

pub fn slugify(phrase: &str) -> String {
    let mut output = String::with_capacity(phrase.len());
    let mut dropped = false;
    for ch in phrase.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
        } else if let Some(translit) = transliterate(ch) {
            output.push_str(translit);
        } else if ch.is_whitespace() || matches!(ch, '-' | '_' | '.' | '\'' | '’') {
            if !output.ends_with('-') {
                output.push('-');
            }
        } else if ch.is_alphanumeric() {
            // A letter we cannot transliterate (CJK, Arabic, ...). Without a
            // hash suffix "hey 你好" and "hey 世界" would share one folder.
            dropped = true;
        }
    }
    let mut slug = output
        .trim_matches('-')
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        return format!("wake-{:08x}", stable_hash(phrase));
    }
    let suffix = dropped.then(|| format!("-{:06x}", stable_hash(phrase) & 0x00ff_ffff));
    let limit = MAX_SLUG_CHARS - suffix.as_ref().map_or(0, String::len);
    if slug.len() > limit {
        slug.truncate(limit);
        slug = slug.trim_end_matches('-').to_string();
    }
    if let Some(suffix) = suffix {
        slug.push_str(&suffix);
    }
    slug
}

fn valid_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug.len() <= MAX_SLUG_CHARS
        && slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !slug.starts_with('-')
        && !slug.ends_with('-')
}

pub fn app_root(app: &AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Cannot resolve app data directory: {error}"))?;
    fs::create_dir_all(&root).map_err(|error| format!("Cannot create app data directory: {error}"))?;
    Ok(root.join("projects"))
}

fn read_project_file(dir: &Path) -> Option<ProjectFile> {
    let text = fs::read_to_string(dir.join(PROJECT_FILE)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_project_file(dir: &Path, project: &ProjectFile) -> Result<(), String> {
    let payload = serde_json::to_string_pretty(project)
        .map_err(|error| format!("Cannot serialize project metadata: {error}"))?;
    fs::write(dir.join(PROJECT_FILE), payload)
        .map_err(|error| format!("Cannot write project metadata: {error}"))
}

/// Resolve an existing project by its slug. Projects are never created
/// implicitly: only `create_project` makes new folders.
pub fn project_dir(app: &AppHandle, slug: &str) -> Result<(ProjectFile, PathBuf), String> {
    if !valid_slug(slug) {
        return Err("Invalid project id.".into());
    }
    let dir = app_root(app)?.join(slug);
    let project = read_project_file(&dir)
        .ok_or_else(|| format!("Project '{slug}' was not found. It may have been deleted or renamed."))?;
    for name in PROJECT_SUBDIRS {
        fs::create_dir_all(dir.join(name))
            .map_err(|error| format!("Cannot create project folder {name}: {error}"))?;
    }
    Ok((project, dir))
}

pub fn recordings_dir(app: &AppHandle, slug: &str) -> Result<PathBuf, String> {
    takes_dir(app, slug, TakeKind::Wake)
}

pub fn takes_dir(app: &AppHandle, slug: &str, kind: TakeKind) -> Result<PathBuf, String> {
    Ok(project_dir(app, slug)?.1.join(kind.folder()))
}

pub fn dataset_dir(app: &AppHandle, slug: &str) -> Result<PathBuf, String> {
    Ok(project_dir(app, slug)?.1.join("dataset"))
}

pub fn models_dir(app: &AppHandle, slug: &str) -> Result<PathBuf, String> {
    Ok(project_dir(app, slug)?.1.join("models"))
}

/// Find a project (other than `except_slug`) that already uses this phrase
/// or would map to the same folder.
fn find_conflict(
    app: &AppHandle,
    phrase: &str,
    slug: &str,
    except_slug: Option<&str>,
) -> Result<Option<String>, String> {
    let wanted = phrase.to_lowercase();
    for item in list_projects(app)? {
        if Some(item.slug.as_str()) == except_slug {
            continue;
        }
        if item.phrase.to_lowercase() == wanted {
            return Ok(Some(format!("A project for \"{}\" already exists.", item.phrase)));
        }
        if item.slug == slug {
            return Ok(Some(format!(
                "Folder \"{slug}\" is already used by \"{}\". Use a more distinct name.",
                item.phrase
            )));
        }
    }
    // A folder without project.json (e.g. a half-deleted project) also blocks the slug.
    let dir = app_root(app)?.join(slug);
    if except_slug != Some(slug) && dir.exists() {
        return Ok(Some(format!("Folder \"{slug}\" already exists in the projects directory.")));
    }
    Ok(None)
}

pub fn preview_project_name(
    app: &AppHandle,
    phrase: &str,
    except_slug: Option<&str>,
) -> Result<ProjectNamePreview, String> {
    let normalized = match normalize_phrase(phrase) {
        Ok(value) => value,
        Err(error) => {
            return Ok(ProjectNamePreview {
                phrase: phrase.trim().to_string(),
                slug: String::new(),
                error: Some(error),
            })
        }
    };
    let slug = slugify(&normalized);
    let error = find_conflict(app, &normalized, &slug, except_slug)?;
    Ok(ProjectNamePreview {
        phrase: normalized,
        slug,
        error,
    })
}

pub fn create_project(app: &AppHandle, phrase: &str) -> Result<ProjectListItem, String> {
    let preview = preview_project_name(app, phrase, None)?;
    if let Some(error) = preview.error {
        return Err(error);
    }
    let dir = app_root(app)?.join(&preview.slug);
    for name in PROJECT_SUBDIRS {
        fs::create_dir_all(dir.join(name))
            .map_err(|error| format!("Cannot create project folder {name}: {error}"))?;
    }
    let now = Utc::now().to_rfc3339();
    write_project_file(
        &dir,
        &ProjectFile {
            phrase: preview.phrase,
            slug: preview.slug.clone(),
            created_at: now.clone(),
            updated_at: now,
        },
    )?;
    project_list_item(&dir).ok_or_else(|| "Cannot read the new project.".into())
}

/// Rename a project. The folder follows the new slug so the projects
/// directory stays readable; recordings, dataset and models move with it.
pub fn rename_project(app: &AppHandle, slug: &str, phrase: &str) -> Result<ProjectListItem, String> {
    let (mut project, dir) = project_dir(app, slug)?;
    let preview = preview_project_name(app, phrase, Some(slug))?;
    if let Some(error) = preview.error {
        return Err(error);
    }
    let target = if preview.slug == slug {
        dir.clone()
    } else {
        let target = app_root(app)?.join(&preview.slug);
        fs::rename(&dir, &target).map_err(|error| {
            format!("Cannot rename the project folder (is a file from it open elsewhere?): {error}")
        })?;
        target
    };
    project.phrase = preview.phrase;
    project.slug = preview.slug;
    project.updated_at = Utc::now().to_rfc3339();
    write_project_file(&target, &project)?;
    project_list_item(&target).ok_or_else(|| "Cannot read the renamed project.".into())
}

pub fn delete_project(app: &AppHandle, slug: &str) -> Result<(), String> {
    let (_, dir) = project_dir(app, slug)?;
    // Remove the marker first: if deletion stops halfway (a locked file on
    // Windows), the leftover folder is no longer listed as a broken project.
    fs::remove_file(dir.join(PROJECT_FILE))
        .map_err(|error| format!("Cannot delete the project: {error}"))?;
    fs::remove_dir_all(&dir).map_err(|error| {
        format!(
            "The project was removed from the list, but some files could not be deleted \
             from {}: {error}",
            dir.display()
        )
    })
}

pub fn new_recording_id() -> String {
    let now = Utc::now();
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    format!(
        "rec_{}_{}",
        now.format("%Y%m%d_%H%M%S"),
        &suffix[..5]
    )
}

fn read_sidecar(path: &Path) -> Option<RecordingSidecar> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn wav_duration_ms(path: &Path) -> Option<u64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return None;
    }
    let frames = u64::from(reader.len()) / u64::from(spec.channels.max(1));
    Some(frames.saturating_mul(1000) / u64::from(spec.sample_rate))
}

fn created_at_from_fs(path: &Path) -> String {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(|_| Utc::now())
        .to_rfc3339()
}

fn list_recordings(dir: &Path, phrase: &str) -> Result<Vec<RecordingMeta>, String> {
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Cannot read recordings folder: {error}")),
    };

    let mut recordings = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let sidecar_path = dir.join(format!("{id}.json"));
        let sidecar = read_sidecar(&sidecar_path);
        let duration_ms = sidecar
            .as_ref()
            .map(|meta| meta.duration_ms)
            .filter(|duration| *duration > 0)
            .or_else(|| wav_duration_ms(&path))
            .unwrap_or(0);
        let size_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        let created_at = sidecar
            .as_ref()
            .map(|meta| meta.created_at.clone())
            .filter(|created| !created.is_empty())
            .unwrap_or_else(|| created_at_from_fs(&path));
        let phrase_value = sidecar
            .as_ref()
            .map(|meta| meta.phrase.clone())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| phrase.to_string());
        recordings.push(RecordingMeta {
            id: id.to_string(),
            file_name: file_name.to_string(),
            duration_ms,
            size_bytes,
            created_at,
            phrase: phrase_value,
            source_sample_rate: sidecar
                .as_ref()
                .map(|meta| meta.source_sample_rate)
                .unwrap_or(0),
            trimmed_ms: sidecar.as_ref().map(|meta| meta.trimmed_ms).unwrap_or(0),
            vad_threshold: sidecar
                .as_ref()
                .map(|meta| meta.vad_threshold)
                .unwrap_or(0.0),
            prompt: sidecar.as_ref().and_then(|meta| meta.prompt.clone()),
        });
    }

    recordings.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(recordings)
}

pub fn save_recording(
    app: &AppHandle,
    slug: &str,
    kind: TakeKind,
    prompt: Option<String>,
    id: &str,
    samples: &[f32],
    source_sample_rate: u32,
    device_name: &str,
    vad_threshold: f32,
    trimmed_ms: u64,
) -> Result<RecordingMeta, String> {
    let (project, project_dir) = project_dir(app, slug)?;
    let phrase = project.phrase;
    let dir = project_dir.join(kind.folder());
    let path = dir.join(format!("{id}.wav"));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)
        .map_err(|error| format!("Cannot create WAV file: {error}"))?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(value)
            .map_err(|error| format!("Cannot write WAV samples: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("Cannot finalize WAV file: {error}"))?;

    let now = Utc::now().to_rfc3339();
    let duration_ms = ((samples.len() as f64 / 16_000.0) * 1000.0).round() as u64;
    let sidecar = RecordingSidecar {
        phrase,
        created_at: now.clone(),
        duration_ms,
        source_sample_rate,
        trimmed_ms,
        vad_threshold,
        prompt: prompt.clone(),
    };
    let sidecar_path = dir.join(format!("{id}.json"));
    let payload = serde_json::to_string_pretty(&sidecar)
        .map_err(|error| format!("Cannot serialize recording metadata: {error}"))?;
    fs::write(&sidecar_path, payload)
        .map_err(|error| format!("Cannot write recording metadata: {error}"))?;

    let size_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
    let _ = device_name; // Captured for future metadata/settings UI.
    Ok(RecordingMeta {
        id: id.to_string(),
        file_name: format!("{id}.wav"),
        duration_ms,
        size_bytes,
        created_at: now,
        phrase: sidecar.phrase,
        source_sample_rate,
        trimmed_ms,
        vad_threshold,
        prompt,
    })
}

fn valid_recording_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 80
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ' '))
        && !id.trim().starts_with('.')
}

pub fn recording_wav_path(
    app: &AppHandle,
    slug: &str,
    kind: TakeKind,
    id: &str,
) -> Result<PathBuf, String> {
    if !valid_recording_id(id) {
        return Err("Invalid recording id.".into());
    }
    let path = takes_dir(app, slug, kind)?.join(format!("{id}.wav"));
    if !path.is_file() {
        return Err("Recording not found.".into());
    }
    Ok(path)
}

pub fn read_recording_bytes(
    app: &AppHandle,
    slug: &str,
    kind: TakeKind,
    id: &str,
) -> Result<Vec<u8>, String> {
    let path = recording_wav_path(app, slug, kind, id)?;
    fs::read(&path).map_err(|error| format!("Cannot read recording: {error}"))
}

pub fn delete_recording(app: &AppHandle, slug: &str, kind: TakeKind, id: &str) -> Result<(), String> {
    let path = recording_wav_path(app, slug, kind, id)?;
    fs::remove_file(&path).map_err(|error| format!("Cannot delete recording: {error}"))?;
    let sidecar = takes_dir(app, slug, kind)?.join(format!("{id}.json"));
    if sidecar.exists() {
        let _ = fs::remove_file(sidecar);
    }
    Ok(())
}

pub fn rename_recording(
    app: &AppHandle,
    slug: &str,
    id: &str,
    new_name: &str,
) -> Result<(), String> {
    let new_name = new_name.trim();
    if !valid_recording_id(new_name) {
        return Err("Name may only contain letters, numbers, spaces, '-' and '_'.".into());
    }
    if new_name.len() > 60 {
        return Err("Recording name is too long (max 60 characters).".into());
    }
    let old_wav = recording_wav_path(app, slug, TakeKind::Wake, id)?;
    let dir = recordings_dir(app, slug)?;
    let new_wav = dir.join(format!("{new_name}.wav"));
    if new_wav.exists() {
        return Err("A recording with that name already exists.".into());
    }
    let old_sidecar = dir.join(format!("{id}.json"));
    let new_sidecar = dir.join(format!("{new_name}.json"));
    fs::rename(&old_wav, &new_wav).map_err(|error| format!("Cannot rename recording: {error}"))?;
    if old_sidecar.exists() {
        let _ = fs::rename(&old_sidecar, &new_sidecar);
    }
    Ok(())
}

fn count_audio(folder: &Path) -> usize {
    fs::read_dir(folder)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry.path().is_file()
                        && entry
                            .path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("wav"))
                            .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

pub fn dataset_summary(project_dir: &Path) -> DatasetSummary {
    let dir = project_dir.join("dataset");
    let info_path = dir.join("dataset_info.json");
    if !info_path.exists() {
        return DatasetSummary::default();
    }
    let info = fs::read_to_string(&info_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .unwrap_or(Value::Null);

    let split = |name: &str| SplitCounts {
        wake_word: count_audio(&dir.join(name).join("wake_word")),
        background: count_audio(&dir.join(name).join("background")),
    };
    let train = split("train");
    let val = split("val");
    let test = split("test");
    let total = train.total() + val.total() + test.total();
    DatasetSummary {
        exists: total > 0,
        train,
        val,
        test,
        total,
        clip_seconds: info.get("clip_seconds").and_then(Value::as_f64),
        variations_per_sample: info.get("variations_per_sample").and_then(Value::as_u64),
        created_at: info
            .get("finished_at")
            .or_else(|| info.get("created_at"))
            .and_then(Value::as_str)
            .map(str::to_string),
        background_source: info
            .get("background_source")
            .and_then(Value::as_str)
            .map(str::to_string),
        warnings: info
            .get("warnings")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

pub fn model_summary(project_dir: &Path) -> ModelSummary {
    let dir = project_dir.join("models");
    let metadata_path = dir.join("metadata.json");
    let tflite_path = dir.join("wake_word_int8.tflite");
    if !metadata_path.exists() || !tflite_path.exists() {
        return ModelSummary::default();
    }
    let metadata = fs::read_to_string(&metadata_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let history = fs::read_to_string(dir.join("history.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());
    let updated_at = metadata
        .as_ref()
        .and_then(|value| value.get("created_at"))
        .and_then(Value::as_str)
        .map(str::to_string);
    ModelSummary {
        exists: true,
        metadata,
        history,
        tflite_size_bytes: fs::metadata(&tflite_path).map(|meta| meta.len()).unwrap_or(0),
        header_available: dir.join("wake_word_model.h").exists(),
        updated_at,
        evaluation: fs::read_to_string(dir.join("evaluation.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok()),
    }
}

pub fn summary(app: &AppHandle, slug: &str) -> Result<ProjectSummary, String> {
    let (project, dir) = project_dir(app, slug)?;
    let phrase = project.phrase;
    let slug = slug.to_string();
    let recordings = list_recordings(&dir.join("recordings"), &phrase)?;
    let negative_takes = list_recordings(&dir.join("negatives"), "")?;
    let dataset = dataset_summary(&dir);
    let model = model_summary(&dir);
    let paths = ProjectPaths {
        project_dir: dir.to_string_lossy().to_string(),
        recordings_dir: dir.join("recordings").to_string_lossy().to_string(),
        negatives_dir: dir.join("negatives").to_string_lossy().to_string(),
        dataset_dir: dir.join("dataset").to_string_lossy().to_string(),
        models_dir: dir.join("models").to_string_lossy().to_string(),
        exports_dir: dir.join("exports").to_string_lossy().to_string(),
    };
    Ok(ProjectSummary {
        phrase,
        slug,
        recordings,
        negative_takes,
        mined_count: count_audio(&dir.join("mined")),
        dataset,
        model,
        paths,
        voice: crate::voice::read_project_voice(&dir),
        synthetic: crate::voice::synthetic_summary(&dir),
    })
}

fn modified_at(path: &Path) -> Option<DateTime<Utc>> {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .map(DateTime::<Utc>::from)
        .ok()
}

fn project_list_item(dir: &Path) -> Option<ProjectListItem> {
    let project = read_project_file(dir)?;
    // The folder name is the project id; it wins over a stale slug in the file.
    let slug = dir.file_name()?.to_str()?.to_string();
    if !valid_slug(&slug) {
        return None;
    }
    // Latest activity: metadata, takes, dataset or model, whichever is newest.
    let updated_at = [
        dir.join(PROJECT_FILE),
        dir.join("recordings"),
        dir.join("dataset").join("dataset_info.json"),
        dir.join("models").join("metadata.json"),
    ]
    .iter()
    .filter_map(|path| modified_at(path))
    .max()
    .map(|date| date.to_rfc3339());
    Some(ProjectListItem {
        phrase: project.phrase,
        slug,
        recording_count: count_audio(&dir.join("recordings")),
        has_dataset: dir.join("dataset").join("dataset_info.json").is_file(),
        has_model: dir.join("models").join("metadata.json").is_file(),
        created_at: Some(project.created_at).filter(|value| !value.is_empty()),
        updated_at,
    })
}

pub fn list_projects(app: &AppHandle) -> Result<Vec<ProjectListItem>, String> {
    let root = app_root(app)?;
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("Cannot read projects: {error}")),
    };
    let mut projects = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|dir| dir.is_dir())
        .filter_map(|dir| project_list_item(&dir))
        .collect::<Vec<_>>();
    projects.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(projects)
}

pub fn export_model(
    app: &AppHandle,
    slug: &str,
    destination: &str,
) -> Result<ExportResult, String> {
    let models = models_dir(app, slug)?;
    let destination = PathBuf::from(destination);
    if !destination.is_dir() {
        return Err("Choose an existing destination folder.".into());
    }

    let source_files = [
        "wake_word_model.h",
        "wake_word_config.h",
        "wake_word_mel.h",
        "wake_word_detector.h",
        "PREPROCESSING.md",
        "wake_word_int8.tflite",
        "metadata.json",
        "history.json",
        "export_info.json",
        "evaluation.json",
    ];
    if !source_files
        .iter()
        .any(|name| models.join(name).is_file())
    {
        return Err("No trained model artifacts found. Train a model first.".into());
    }

    let mut copied = Vec::new();
    for name in source_files {
        let source = models.join(name);
        if !source.is_file() {
            continue;
        }
        let target = destination.join(name);
        fs::copy(&source, &target)
            .map_err(|error| format!("Cannot export {name}: {error}"))?;
        copied.push(target.to_string_lossy().to_string());
    }

    if copied.iter().any(|path| path.ends_with("wake_word_model.h")) {
        Ok(ExportResult {
            destination: destination.to_string_lossy().to_string(),
            files: copied,
        })
    } else {
        Err("The C header was not found. Retrain the model.".into())
    }
}

/// Clips written by the live test (python/live.py).
pub fn live_dir(app: &AppHandle, slug: &str) -> Result<PathBuf, String> {
    Ok(project_dir(app, slug)?.1.join("live"))
}

fn live_clip_path(app: &AppHandle, slug: &str, file: &str) -> Result<PathBuf, String> {
    let valid = file.ends_with(".wav")
        && (file.starts_with("detection_") || file.starts_with("missed_"))
        && file.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        && !file.contains("..");
    if !valid {
        return Err("Invalid live clip name.".into());
    }
    let path = live_dir(app, slug)?.join(file);
    if !path.is_file() {
        return Err("This clip is no longer available.".into());
    }
    Ok(path)
}

pub fn read_live_clip(app: &AppHandle, slug: &str, file: &str) -> Result<Vec<u8>, String> {
    fs::read(live_clip_path(app, slug, file)?).map_err(|error| format!("Cannot read clip: {error}"))
}

/// Keep a live clip as training data: a take of the wake phrase, or other speech
/// that must not trigger (a false detection).
pub fn save_live_clip(app: &AppHandle, slug: &str, file: &str, kind: TakeKind) -> Result<RecordingMeta, String> {
    let path = live_clip_path(app, slug, file)?;
    let mut reader = hound::WavReader::open(&path).map_err(|error| format!("Cannot open clip: {error}"))?;
    let rate = reader.spec().sample_rate;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|value| value as f32 / 32768.0)
        .collect();
    if rate != 16_000 || samples.is_empty() {
        return Err("The clip is not 16 kHz audio.".into());
    }
    match kind {
        TakeKind::Negative => {
            let prompt = Some(format!("Live test: false detection ({file})"));
            save_recording(app, slug, kind, prompt, &new_recording_id(), &samples, rate, "live test", 0.0, 0)
        }
        TakeKind::Wake => {
            // A detection clip holds 2.0 s before and 0.6 s after the detection, so
            // other speech may precede the phrase: keep 1.5 s before to 0.4 s after
            // the detection. Snapshots ("missed") are kept whole. Then trim silence
            // with the recorder's VAD, like a take recorded in the app.
            let segment = if file.starts_with("detection_") {
                let second = 16_000usize;
                let end = samples.len().saturating_sub(second / 5);
                let start = samples.len().saturating_sub(second * 21 / 10);
                samples[start..end].to_vec()
            } else {
                samples
            };
            let capture = crate::audio::CaptureResult {
                samples: segment,
                input_sample_rate: rate,
                device_name: "live test".into(),
                elapsed_ms: 0,
            };
            let processed = crate::audio::process_capture(&capture)?;
            save_recording(
                app,
                slug,
                kind,
                None,
                &new_recording_id(),
                &processed.samples,
                rate,
                "live test",
                processed.report.threshold,
                processed.report.trimmed_ms,
            )
        }
    }
}

pub fn delete_mined(app: &AppHandle, slug: &str) -> Result<(), String> {
    let dir = project_dir(app, slug)?.1.join("mined");
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|error| format!("Cannot delete mined negatives: {error}"))?;
    }
    Ok(())
}

pub fn ensure_root(app: &AppHandle) -> Result<(), String> {
    app_root(app).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_are_stable_for_latin_and_cyrillic() {
        assert_eq!(slugify("Hey Jarvis"), "hey-jarvis");
        assert_eq!(slugify("Среда"), "sreda");
        assert_eq!(slugify("  ok,   google! "), "ok-google");
        assert!(valid_slug(&slugify("Среда")));
    }

    #[test]
    fn untransliterated_letters_do_not_collide() {
        let first = slugify("hey 你好");
        let second = slugify("hey 世界");
        assert_ne!(first, second);
        assert!(first.starts_with("hey-"));
        assert!(valid_slug(&first) && valid_slug(&second));
        assert!(slugify("你好").starts_with("wake-"));
    }

    #[test]
    fn long_names_keep_the_hash_suffix() {
        let slug = slugify(&format!("{} 你好", "a".repeat(80)));
        assert!(slug.len() <= MAX_SLUG_CHARS);
        assert!(valid_slug(&slug));
    }

    #[test]
    fn take_kinds_map_to_separate_folders() {
        assert_eq!(TakeKind::parse(None).unwrap(), TakeKind::Wake);
        assert_eq!(TakeKind::parse(Some("negative")).unwrap().folder(), "negatives");
        assert_eq!(TakeKind::parse(Some("wake")).unwrap().folder(), "recordings");
        assert!(TakeKind::parse(Some("../models")).is_err());
    }
}
