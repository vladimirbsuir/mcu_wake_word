mod audio;
mod project;
mod python_bridge;
mod voice;

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager, State};

use audio::{CaptureSession, CaptureStart, Meter, TrimReport};
use project::{
    ExportResult, ProjectListItem, ProjectNamePreview, ProjectSummary, RecordingMeta, TakeKind,
};
use python_bridge::{EnvironmentStatus, JobStarted};
use voice::VoiceSettings;

/// A capture remembers the project it was started for, so a take can never be
/// saved into a different project than the one shown when Record was pressed.
struct ActiveCapture {
    project: String,
    kind: TakeKind,
    prompt: Option<String>,
    session: CaptureSession,
}

pub struct AppState {
    capture: Mutex<Option<ActiveCapture>>,
    jobs: python_bridge::JobState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            capture: Mutex::new(None),
            jobs: python_bridge::JobState::default(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StopRecordingResult {
    recording: RecordingMeta,
    trim: TrimReport,
    capture_ms: u64,
    output_sample_rate: u32,
}

fn ensure_no_active_recording(state: &AppState) -> Result<(), String> {
    let capture = state
        .capture
        .lock()
        .map_err(|_| "Microphone state is unavailable.".to_string())?;
    if capture.is_some() {
        return Err("Stop the active recording before starting a pipeline job.".into());
    }
    Ok(())
}

fn ensure_no_active_job(state: &AppState) -> Result<(), String> {
    let jobs = state
        .jobs
        .current
        .lock()
        .map_err(|_| "Job manager is unavailable.".to_string())?;
    if let Some(job) = jobs.as_ref() {
        return Err(format!(
            "Wait for the current {} job to finish first.",
            job.kind
        ));
    }
    Ok(())
}

/// Renaming or deleting moves/removes folders that a recording or a Python
/// job may be writing to, so both require a fully idle app.
fn ensure_idle(state: &AppState, action: &str) -> Result<(), String> {
    let recording = state
        .capture
        .lock()
        .map_err(|_| "Microphone state is unavailable.".to_string())?
        .is_some();
    if recording {
        return Err(format!("Stop the active recording before you {action}."));
    }
    let job = state
        .jobs
        .current
        .lock()
        .map_err(|_| "Job manager is unavailable.".to_string())?
        .as_ref()
        .map(|job| job.kind.clone());
    if let Some(kind) = job {
        return Err(format!("Wait for the {kind} job to finish before you {action}."));
    }
    Ok(())
}

#[tauri::command]
fn get_project_summary(app: AppHandle, project: String) -> Result<ProjectSummary, String> {
    project::summary(&app, &project)
}

#[tauri::command]
fn list_projects(app: AppHandle) -> Result<Vec<ProjectListItem>, String> {
    project::list_projects(&app)
}

#[tauri::command]
fn preview_project_name(
    app: AppHandle,
    phrase: String,
    except_project: Option<String>,
) -> Result<ProjectNamePreview, String> {
    project::preview_project_name(&app, &phrase, except_project.as_deref())
}

#[tauri::command]
fn create_project(app: AppHandle, phrase: String) -> Result<ProjectListItem, String> {
    project::create_project(&app, &phrase)
}

#[tauri::command]
fn rename_project(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    phrase: String,
) -> Result<ProjectListItem, String> {
    ensure_idle(&state, "rename a project")?;
    project::rename_project(&app, &project, &phrase)
}

#[tauri::command]
fn delete_project(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<(), String> {
    ensure_idle(&state, "delete a project")?;
    project::delete_project(&app, &project)
}

#[tauri::command]
fn start_recording(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    kind: Option<String>,
    prompt: Option<String>,
) -> Result<CaptureStart, String> {
    let kind = TakeKind::parse(kind.as_deref())?;
    let prompt = prompt
        .map(|text| text.trim().chars().take(400).collect::<String>())
        .filter(|text| !text.is_empty());
    // Validates the project and creates its take folders.
    project::takes_dir(&app, &project, kind)?;
    if let Ok(jobs) = state.jobs.current.lock() {
        // Global jobs such as model downloads do not use audio or project files.
        if jobs.as_ref().is_some_and(|job| !job.project.is_empty()) {
            return Err("Wait for the current dataset/training job to finish first.".into());
        }
    }
    let mut capture_slot = state
        .capture
        .lock()
        .map_err(|_| "Microphone state is unavailable.".to_string())?;
    if capture_slot.is_some() {
        return Err("A recording is already in progress.".into());
    }
    let id = project::new_recording_id();
    let session = CaptureSession::start(id)?;
    let info = session.start_info();
    *capture_slot = Some(ActiveCapture {
        project,
        kind,
        prompt,
        session,
    });
    Ok(info)
}

#[tauri::command]
fn get_recording_meter(state: State<'_, AppState>) -> Result<Meter, String> {
    let capture_slot = state
        .capture
        .lock()
        .map_err(|_| "Microphone state is unavailable.".to_string())?;
    Ok(capture_slot
        .as_ref()
        .map(|capture| capture.session.meter())
        .unwrap_or_default())
}

#[tauri::command]
fn stop_recording(app: AppHandle, state: State<'_, AppState>) -> Result<StopRecordingResult, String> {
    let ActiveCapture {
        project,
        kind,
        prompt,
        session,
    } = {
        let mut slot = state
            .capture
            .lock()
            .map_err(|_| "Microphone state is unavailable.".to_string())?;
        slot.take()
            .ok_or_else(|| "There is no active recording.".to_string())?
    };
    let id = session.id.clone();
    let capture = session.stop()?;
    let processed = audio::process_capture(&capture)?;
    let recording = project::save_recording(
        &app,
        &project,
        kind,
        prompt,
        &id,
        &processed.samples,
        capture.input_sample_rate,
        &capture.device_name,
        processed.report.threshold,
        processed.report.trimmed_ms,
    )?;
    Ok(StopRecordingResult {
        recording,
        trim: processed.report,
        capture_ms: capture.elapsed_ms,
        output_sample_rate: audio::TARGET_SAMPLE_RATE,
    })
}

#[tauri::command]
fn cancel_recording(state: State<'_, AppState>) -> Result<(), String> {
    let mut slot = state
        .capture
        .lock()
        .map_err(|_| "Microphone state is unavailable.".to_string())?;
    if let Some(capture) = slot.take() {
        capture.session.cancel();
    }
    Ok(())
}

#[tauri::command]
fn read_recording_audio(
    app: AppHandle,
    project: String,
    id: String,
    kind: Option<String>,
) -> Result<Vec<u8>, String> {
    project::read_recording_bytes(&app, &project, TakeKind::parse(kind.as_deref())?, &id)
}

#[tauri::command]
fn delete_recording(
    app: AppHandle,
    project: String,
    id: String,
    kind: Option<String>,
) -> Result<(), String> {
    project::delete_recording(&app, &project, TakeKind::parse(kind.as_deref())?, &id)
}

#[tauri::command]
fn rename_recording(
    app: AppHandle,
    project: String,
    id: String,
    new_name: String,
) -> Result<(), String> {
    project::rename_recording(&app, &project, &id, &new_name)
}

#[tauri::command]
fn get_environment(app: AppHandle) -> EnvironmentStatus {
    python_bridge::environment_status(&app)
}

#[tauri::command]
fn start_augmentation(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    variations: u32,
    clip_seconds: f64,
    background_dir: Option<String>,
    negative_dir: Option<String>,
    use_synthetic: Option<bool>,
    use_corpora: Option<bool>,
    corpus_negatives: Option<u32>,
    use_mined: Option<bool>,
) -> Result<JobStarted, String> {
    ensure_no_active_recording(&state)?;
    if !(10..=40).contains(&variations) {
        return Err("Variations per recording must be between 10 and 40.".into());
    }
    if !(1.0..=4.0).contains(&clip_seconds) {
        return Err("Clip length must be between 1.0 and 4.0 seconds.".into());
    }
    let summary = project::summary(&app, &project)?;
    if summary.recordings.len() < 3 {
        return Err(
            "Record at least 3 clean takes first. 5-10 takes give a much better model.".into(),
        );
    }

    let recordings_dir = project::recordings_dir(&app, &project)?;
    let dataset_dir = project::dataset_dir(&app, &project)?;
    if let Some(folder) = background_dir.as_ref() {
        let path = std::path::PathBuf::from(folder);
        if !path.is_dir() {
            return Err("The selected background folder does not exist.".into());
        }
    }
    if let Some(folder) = negative_dir.as_ref() {
        if !std::path::Path::new(folder).is_dir() {
            return Err("The selected negative phrase folder does not exist.".into());
        }
    }

    let mut args = vec![
        "--input".into(),
        recordings_dir.to_string_lossy().to_string(),
        "--output".into(),
        dataset_dir.to_string_lossy().to_string(),
        "--phrase".into(),
        voice::spoken_phrase(&project::project_dir(&app, &project)?.1, &summary.phrase),
        "--variations".into(),
        variations.to_string(),
        "--clip-seconds".into(),
        format!("{clip_seconds}"),
        "--overwrite".into(),
    ];
    if let Some(folder) = background_dir {
        args.push("--background-dir".into());
        args.push(folder);
    }
    if let Some(folder) = negative_dir {
        args.push("--negative-dir".into());
        args.push(folder);
    }
    // The speaker's own "other speech" takes are always hard negatives.
    if !summary.negative_takes.is_empty() {
        args.push("--negative-dir".into());
        args.push(project::takes_dir(&app, &project, TakeKind::Negative)?.to_string_lossy().to_string());
    }
    let synthetic = voice::synthetic_dir(&project::project_dir(&app, &project)?.1);
    if use_synthetic.unwrap_or(true) && synthetic.join("manifest.json").is_file() {
        args.push("--synthetic-dir".into());
        args.push(synthetic.to_string_lossy().to_string());
    }
    if use_mined.unwrap_or(true) && summary.mined_count > 0 {
        args.push("--mined-dir".into());
        args.push(project::project_dir(&app, &project)?.1.join("mined").to_string_lossy().to_string());
    }
    if use_corpora.unwrap_or(true) {
        for folder in voice::installed_corpora(&app, "negatives")? {
            args.push("--speech-corpus".into());
            args.push(folder.to_string_lossy().to_string());
        }
        if let Some(count) = corpus_negatives {
            if !(500..=40_000).contains(&count) {
                return Err("Speech clips must be between 500 and 40000.".into());
            }
            args.push("--corpus-negatives".into());
            args.push(count.to_string());
        }
    }

    python_bridge::start_job(app, &state.jobs, &project, "augment", "augment.py", args, Vec::new())
}

#[tauri::command]
fn start_training(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    architecture: Option<String>,
) -> Result<JobStarted, String> {
    ensure_no_active_recording(&state)?;
    let architecture = architecture.unwrap_or_else(|| "ds-cnn-temporal".into());
    if !matches!(architecture.as_str(), "ds-cnn" | "ds-cnn-temporal") {
        return Err("Unknown model architecture.".into());
    }
    if !(1..=300).contains(&epochs) {
        return Err("Epochs must be between 1 and 300.".into());
    }
    if !(1..=512).contains(&batch_size) {
        return Err("Batch size must be between 1 and 512.".into());
    }
    if learning_rate <= 0.0 || learning_rate > 1.0 {
        return Err("Learning rate must be between 0 and 1.".into());
    }

    let summary = project::summary(&app, &project)?;
    if !summary.dataset.exists || summary.dataset.train.total() == 0 {
        return Err("Build an augmented dataset before starting training.".into());
    }
    let dataset_dir = project::dataset_dir(&app, &project)?;
    let models_dir = project::models_dir(&app, &project)?;
    let clip_seconds = summary.dataset.clip_seconds.unwrap_or(1.5);

    let args = vec![
        "--architecture".into(),
        architecture,
        "--dataset".into(),
        dataset_dir.to_string_lossy().to_string(),
        "--output".into(),
        models_dir.to_string_lossy().to_string(),
        "--phrase".into(),
        voice::spoken_phrase(&project::project_dir(&app, &project)?.1, &summary.phrase),
        "--epochs".into(),
        epochs.to_string(),
        "--batch-size".into(),
        batch_size.to_string(),
        "--learning-rate".into(),
        format!("{learning_rate}"),
        "--clip-seconds".into(),
        format!("{clip_seconds}"),
    ];
    python_bridge::start_job(app, &state.jobs, &project, "train", "train.py", args, Vec::new())
}

/// Stream benchmark speech and the held-out takes through the exported model.
#[tauri::command]
fn start_evaluation(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<JobStarted, String> {
    ensure_no_active_recording(&state)?;
    let (project_file, dir) = project::project_dir(&app, &project)?;
    if !dir.join("models").join("wake_word_int8.tflite").is_file() {
        return Err("Train a model first.".into());
    }
    let corpora = voice::installed_corpora(&app, "eval")?;
    if corpora.is_empty() {
        return Err("Download a benchmark speech set first (AI models → Speech data).".into());
    }
    let spoken = voice::spoken_phrase(&dir, &project_file.phrase);
    let mut args = vec![
        "--models".into(),
        dir.join("models").to_string_lossy().to_string(),
        "--phrase".into(),
        spoken,
    ];
    for folder in corpora {
        args.push("--corpus".into());
        args.push(folder.to_string_lossy().to_string());
    }
    if dir.join("dataset").join("dataset_info.json").is_file() {
        args.extend([
            "--dataset".into(),
            dir.join("dataset").to_string_lossy().to_string(),
            "--recordings".into(),
            dir.join("recordings").to_string_lossy().to_string(),
        ]);
    }
    python_bridge::start_job(app, &state.jobs, &project, "evaluate", "evaluate.py", args, Vec::new())
}

/// Find training speech the current model fires on and save it as hard negatives.
#[tauri::command]
fn start_mining(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<JobStarted, String> {
    ensure_no_active_recording(&state)?;
    let (project_file, dir) = project::project_dir(&app, &project)?;
    if !dir.join("models").join("wake_word_int8.tflite").is_file() {
        return Err("Train a model first.".into());
    }
    let corpora = voice::installed_corpora(&app, "negatives")?;
    if corpora.is_empty() {
        return Err("Download a training speech set first (AI models → Speech data).".into());
    }
    let mut args = vec![
        "--models".into(),
        dir.join("models").to_string_lossy().to_string(),
        "--output".into(),
        dir.join("mined").to_string_lossy().to_string(),
        "--phrase".into(),
        voice::spoken_phrase(&dir, &project_file.phrase),
    ];
    for folder in corpora {
        args.push("--corpus".into());
        args.push(folder.to_string_lossy().to_string());
    }
    python_bridge::start_job(app, &state.jobs, &project, "mine", "mine.py", args, Vec::new())
}

#[tauri::command]
fn delete_mined(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<(), String> {
    ensure_no_active_job(&state)?;
    project::delete_mined(&app, &project)
}

/// Run the trained model on the PC microphone until the job is canceled.
#[tauri::command]
fn start_live(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    rule: String,
    windows: u32,
    threshold: f64,
    refractory_ms: u32,
    gain_db: f64,
) -> Result<JobStarted, String> {
    ensure_no_active_recording(&state)?;
    let dir = project::project_dir(&app, &project)?.1;
    if !dir.join("models").join("wake_word_int8.tflite").is_file() {
        return Err("Train a model first.".into());
    }
    if !matches!(rule.as_str(), "all" | "mean") || !(1..=8).contains(&windows) {
        return Err("Unknown detector rule.".into());
    }
    if !(0.0..=1.0).contains(&threshold) || !(-20.0..=40.0).contains(&gain_db) || refractory_ms > 10_000 {
        return Err("Detector settings are out of range.".into());
    }
    let args = vec![
        "--models".into(),
        dir.join("models").to_string_lossy().to_string(),
        "--clips-dir".into(),
        project::live_dir(&app, &project)?.to_string_lossy().to_string(),
        "--rule".into(),
        rule,
        "--windows".into(),
        windows.to_string(),
        "--threshold".into(),
        format!("{threshold}"),
        "--refractory-ms".into(),
        refractory_ms.to_string(),
        "--gain-db".into(),
        format!("{gain_db}"),
    ];
    python_bridge::start_job(app, &state.jobs, &project, "live", "live.py", args, Vec::new())
}

/// Ask the running live test to save the last seconds ("it missed my phrase").
#[tauri::command]
fn live_snapshot(app: AppHandle, project: String) -> Result<(), String> {
    let dir = project::live_dir(&app, &project)?;
    std::fs::create_dir_all(&dir).map_err(|error| format!("Cannot create live folder: {error}"))?;
    std::fs::write(dir.join(".snap"), b"").map_err(|error| format!("Cannot request a snapshot: {error}"))
}

#[tauri::command]
fn read_live_clip(app: AppHandle, project: String, file: String) -> Result<Vec<u8>, String> {
    project::read_live_clip(&app, &project, &file)
}

#[tauri::command]
fn save_live_clip(app: AppHandle, project: String, file: String, kind: String) -> Result<RecordingMeta, String> {
    project::save_live_clip(&app, &project, &file, TakeKind::parse(Some(kind.as_str()))?)
}

#[tauri::command]
fn cancel_job(app: AppHandle) -> Result<(), String> {
    python_bridge::cancel_job(&app)
}

#[tauri::command]
fn export_for_esp32(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    destination: String,
) -> Result<ExportResult, String> {
    ensure_no_active_recording(&state)?;
    ensure_no_active_job(&state)?;
    project::export_model(&app, &project, &destination)
}

// ------------------------------------------------------------ AI providers

/// Global (not project-bound) jobs such as model downloads use this id.
const GLOBAL_JOB: &str = "";

fn voice_args(app: &AppHandle, command: &str) -> Result<Vec<String>, String> {
    let mut args = vec![
        command.to_string(),
        "--models-dir".into(),
        voice::models_dir(app)?.to_string_lossy().to_string(),
    ];
    if matches!(command, "status" | "corpus") {
        args.push("--datasets-dir".into());
        args.push(voice::datasets_dir(app)?.to_string_lossy().to_string());
    }
    if !matches!(command, "status" | "install" | "deps" | "corpus") {
        args.push("--settings".into());
        args.push(voice::settings_path(app)?.to_string_lossy().to_string());
    }
    Ok(args)
}

fn start_voice_job(
    app: AppHandle,
    state: &AppState,
    project: &str,
    kind: &str,
    args: Vec<String>,
) -> Result<JobStarted, String> {
    if !project.is_empty() {
        ensure_no_active_recording(state)?;
    }
    let envs = voice::job_env(&app)?;
    python_bridge::start_job(app, &state.jobs, project, kind, "voice.py", args, envs)
}

#[tauri::command]
async fn voice_status(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let args = voice_args(&app, "status")?;
        python_bridge::run_script(&app, "voice.py", &args)
    })
    .await
    .map_err(|error| format!("Model catalog query failed: {error}"))?
}

#[tauri::command]
fn voice_settings_get(app: AppHandle) -> Result<VoiceSettings, String> {
    voice::load_settings(&app)
}

#[tauri::command]
fn voice_settings_save(app: AppHandle, settings: VoiceSettings) -> Result<VoiceSettings, String> {
    voice::save_settings(&app, settings)
}

#[tauri::command]
fn voice_set_api_key(app: AppHandle, profile_id: String, key: Option<String>) -> Result<VoiceSettings, String> {
    voice::set_api_key(&profile_id, key.as_deref())?;
    voice::load_settings(&app)
}

#[tauri::command]
fn voice_install_model(app: AppHandle, state: State<'_, AppState>, model: String) -> Result<JobStarted, String> {
    if !voice::valid_model_id(&model) {
        return Err("Invalid model id.".into());
    }
    let mut args = voice_args(&app, "install")?;
    args.extend(["--model".into(), model]);
    start_voice_job(app, &state, GLOBAL_JOB, "install", args)
}

#[tauri::command]
fn voice_install_packages(app: AppHandle, state: State<'_, AppState>, model: String) -> Result<JobStarted, String> {
    if !voice::valid_model_id(&model) {
        return Err("Invalid model id.".into());
    }
    let mut args = voice_args(&app, "deps")?;
    args.extend(["--model".into(), model]);
    start_voice_job(app, &state, GLOBAL_JOB, "packages", args)
}

#[tauri::command]
fn voice_install_corpus(app: AppHandle, state: State<'_, AppState>, corpus: String) -> Result<JobStarted, String> {
    if !voice::valid_model_id(&corpus) {
        return Err("Invalid corpus id.".into());
    }
    let mut args = voice_args(&app, "corpus")?;
    args.extend(["--corpus".into(), corpus]);
    start_voice_job(app, &state, GLOBAL_JOB, "corpus", args)
}

#[tauri::command]
fn voice_remove_corpus(app: AppHandle, state: State<'_, AppState>, corpus: String) -> Result<(), String> {
    ensure_no_active_job(&state)?;
    voice::remove_corpus(&app, &corpus)
}

#[tauri::command]
fn voice_remove_model(app: AppHandle, state: State<'_, AppState>, model: String) -> Result<(), String> {
    ensure_no_active_job(&state)?;
    voice::remove_model(&app, &model)
}

#[tauri::command]
fn voice_check(
    app: AppHandle,
    state: State<'_, AppState>,
    category: String,
    language: Option<String>,
) -> Result<JobStarted, String> {
    if !voice::CATEGORIES.contains(&category.as_str()) {
        return Err("Unknown category.".into());
    }
    let mut args = voice_args(&app, "check")?;
    args.extend(["--category".into(), category]);
    if let Some(language) = language.filter(|value| !value.is_empty()) {
        args.extend(["--language".into(), language]);
    }
    start_voice_job(app, &state, GLOBAL_JOB, "check", args)
}

#[tauri::command]
fn voice_project_save(app: AppHandle, project: String, data: Value) -> Result<(), String> {
    let (_, dir) = project::project_dir(&app, &project)?;
    voice::write_project_voice(&dir, &data)
}

fn project_language(dir: &std::path::Path) -> Result<String, String> {
    voice::read_project_voice(dir)
        .get("language")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "Choose the language of the wake phrase first.".to_string())
}

#[tauri::command]
fn voice_transcribe(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<JobStarted, String> {
    let (_, dir) = project::project_dir(&app, &project)?;
    let mut args = voice_args(&app, "transcribe")?;
    args.extend([
        "--recordings".into(),
        dir.join("recordings").to_string_lossy().to_string(),
        "--language".into(),
        project_language(&dir)?,
    ]);
    start_voice_job(app, &state, &project, "transcribe", args)
}

#[tauri::command]
fn voice_generate_words(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    groups: BTreeMap<String, u32>,
) -> Result<JobStarted, String> {
    let (_, dir) = project::project_dir(&app, &project)?;
    project_language(&dir)?;
    if groups.values().all(|count| *count == 0) || groups.values().any(|count| *count > 200) {
        return Err("Ask for 1-200 items per group.".into());
    }
    let mut args = voice_args(&app, "words")?;
    args.extend([
        "--voice-file".into(),
        voice::voice_file(&dir).to_string_lossy().to_string(),
        "--groups".into(),
        serde_json::to_string(&groups).map_err(|error| error.to_string())?,
    ]);
    start_voice_job(app, &state, &project, "words", args)
}

#[tauri::command]
fn voice_synthesize(
    app: AppHandle,
    state: State<'_, AppState>,
    project: String,
    include_positives: bool,
    voices_per_negative: u32,
) -> Result<JobStarted, String> {
    let (_, dir) = project::project_dir(&app, &project)?;
    project_language(&dir)?;
    if !(1..=5).contains(&voices_per_negative) {
        return Err("Use 1-5 voices per word.".into());
    }
    let mut args = voice_args(&app, "synthesize")?;
    args.extend([
        "--voice-file".into(),
        voice::voice_file(&dir).to_string_lossy().to_string(),
        "--output".into(),
        voice::synthetic_dir(&dir).to_string_lossy().to_string(),
        "--voices-per-negative".into(),
        voices_per_negative.to_string(),
        if include_positives { "--positives".into() } else { "--no-positives".into() },
    ]);
    start_voice_job(app, &state, &project, "synthesize", args)
}

#[tauri::command]
fn read_synthetic_audio(app: AppHandle, project: String, file: String) -> Result<Vec<u8>, String> {
    let (_, dir) = project::project_dir(&app, &project)?;
    voice::read_synthetic_clip(&dir, &file)
}

#[tauri::command]
fn delete_synthetic(app: AppHandle, state: State<'_, AppState>, project: String) -> Result<(), String> {
    ensure_no_active_job(&state)?;
    let (_, dir) = project::project_dir(&app, &project)?;
    voice::delete_synthetic(&dir)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            project::ensure_root(app.handle())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_project_summary,
            list_projects,
            preview_project_name,
            create_project,
            rename_project,
            delete_project,
            start_recording,
            get_recording_meter,
            stop_recording,
            cancel_recording,
            read_recording_audio,
            delete_recording,
            rename_recording,
            get_environment,
            start_augmentation,
            start_training,
            start_evaluation,
            start_mining,
            delete_mined,
            start_live,
            live_snapshot,
            read_live_clip,
            save_live_clip,
            cancel_job,
            export_for_esp32,
            voice_status,
            voice_settings_get,
            voice_settings_save,
            voice_set_api_key,
            voice_install_model,
            voice_install_packages,
            voice_remove_model,
            voice_install_corpus,
            voice_remove_corpus,
            voice_check,
            voice_project_save,
            voice_transcribe,
            voice_generate_words,
            voice_synthesize,
            read_synthetic_audio,
            delete_synthetic,
        ])
        .build(tauri::generate_context!())
        .expect("error while building WakeForge Studio");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            // Do not leave a TensorFlow process running after the window closes.
            if let Some(state) = app_handle.try_state::<AppState>() {
                if let Ok(slot) = state.jobs.current.lock() {
                    if let Some(job) = slot.as_ref() {
                        if let Ok(mut child) = job.child.lock() {
                            if let Some(child) = child.as_mut() {
                                let _ = child.kill();
                            }
                        }
                    }
                }
            }
            if let Ok(mut slot) = app_handle.state::<AppState>().capture.lock() {
                if let Some(capture) = slot.take() {
                    capture.session.cancel();
                }
            }
        }
    });
}
