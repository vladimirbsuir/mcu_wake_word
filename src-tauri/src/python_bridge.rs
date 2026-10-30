//! Python process orchestration.
//!
//! The frontend never talks to Python directly: commands spawn a script, log
//! stdout/stderr as Tauri events, parse machine-readable progress lines, and
//! return the final result payload. Only one heavy job can run at a time.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

const PROGRESS_PREFIX: &str = "@@PROGRESS@@";
const RESULT_PREFIX: &str = "@@RESULT@@";

#[derive(Default)]
pub struct JobState {
    pub current: Mutex<Option<ActiveJob>>,
}

pub struct ActiveJob {
    pub id: String,
    /// Empty for global jobs (model downloads) that touch no project folder.
    pub project: String,
    pub kind: String,
    pub child: Arc<Mutex<Option<Child>>>,
    pub canceled: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStarted {
    pub job_id: String,
    pub project: String,
    pub kind: String,
    pub python: String,
    pub script: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobLogEvent {
    pub job_id: String,
    pub project: String,
    pub kind: String,
    pub stream: String,
    pub line: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobFinishedEvent {
    pub job_id: String,
    pub project: String,
    pub kind: String,
    pub success: bool,
    pub canceled: bool,
    pub exit_code: Option<i32>,
    pub result: Option<Value>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentStatus {
    pub ready: bool,
    pub python: String,
    pub executable: String,
    pub version: String,
    pub supported_python: bool,
    pub packages: BTreeMap<String, bool>,
    pub message: String,
    pub script_dir: String,
}

#[derive(Debug, Deserialize)]
struct EnvProbe {
    executable: String,
    version: String,
    supported_python: bool,
    packages: BTreeMap<String, bool>,
    all_packages_present: bool,
}

pub fn python_scripts_dir(app: &AppHandle) -> Result<PathBuf, String> {
    // Development: keep using the checkout even when Vite changes cwd.
    let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../python");
    if dev.join("train.py").is_file() {
        return dev
            .canonicalize()
            .map_err(|error| format!("Cannot resolve python folder: {error}"));
    }
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Cannot resolve resource directory: {error}"))?;
    let packaged = resource_dir.join("python");
    if packaged.join("train.py").is_file() {
        return Ok(packaged);
    }
    Err("Python pipeline scripts were not found. Reinstall WakeForge Studio.".into())
}

fn command_for(python: &Path) -> Command {
    let mut command = Command::new(python);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

fn is_working_python(path: &Path) -> bool {
    command_for(path)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub fn find_python(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(override_path) = std::env::var("WAKE_WORD_PYTHON") {
        let path = PathBuf::from(&override_path);
        if path.is_file() {
            return Ok(path);
        }
        // Allow a bare command such as "python3" as well.
        let candidate = PathBuf::from(&override_path);
        if candidate.components().count() == 1 && is_working_python(&candidate) {
            return Ok(candidate);
        }
        return Err(format!(
            "WAKE_WORD_PYTHON is set to '{override_path}', but that interpreter could not be used."
        ));
    }

    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut candidates = vec![
        project_root.join(".venv/Scripts/python.exe"),
        project_root.join(".venv/bin/python"),
        project_root.join("python/.venv/Scripts/python.exe"),
        project_root.join("python/.venv/bin/python"),
    ];

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join(".venv/Scripts/python.exe"));
        candidates.push(resource_dir.join(".venv/bin/python"));
    }

    for candidate in candidates {
        if candidate.is_file() && is_working_python(&candidate) {
            return Ok(candidate);
        }
    }

    #[cfg(windows)]
    let fallbacks = ["python.exe", "python"];
    #[cfg(not(windows))]
    let fallbacks = ["python3", "python"];

    for fallback in fallbacks {
        let path = PathBuf::from(fallback);
        if is_working_python(&path) {
            return Ok(path);
        }
    }
    Err(
        "Python 3.10-3.12 was not found. Create .venv and install python/requirements.txt, \
         or set WAKE_WORD_PYTHON to a Python executable."
            .into(),
    )
}

pub fn environment_status(app: &AppHandle) -> EnvironmentStatus {
    let scripts = match python_scripts_dir(app) {
        Ok(path) => path,
        Err(error) => {
            return EnvironmentStatus {
                ready: false,
                python: String::new(),
                executable: String::new(),
                version: String::new(),
                supported_python: false,
                packages: BTreeMap::new(),
                message: error,
                script_dir: String::new(),
            }
        }
    };
    let python = match find_python(app) {
        Ok(path) => path,
        Err(error) => {
            return EnvironmentStatus {
                ready: false,
                python: String::new(),
                executable: String::new(),
                version: String::new(),
                supported_python: false,
                packages: BTreeMap::new(),
                message: error,
                script_dir: scripts.to_string_lossy().to_string(),
            }
        }
    };

    let output = command_for(&python)
        .arg(scripts.join("check_env.py"))
        .env("PYTHONIOENCODING", "utf-8")
        .output();

    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return EnvironmentStatus {
                ready: false,
                python: python.to_string_lossy().to_string(),
                executable: String::new(),
                version: String::new(),
                supported_python: false,
                packages: BTreeMap::new(),
                message: format!("Cannot launch Python: {error}"),
                script_dir: scripts.to_string_lossy().to_string(),
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let probe = stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<EnvProbe>(line.trim()).ok());

    match probe {
        Some(probe) if output.status.success() => {
            let ready = probe.supported_python && probe.all_packages_present;
            let mut missing = probe
                .packages
                .iter()
                .filter_map(|(name, present)| (!present).then_some(name.as_str()))
                .collect::<Vec<_>>();
            missing.sort_unstable();
            let message = if !probe.supported_python {
                format!(
                    "Python {} is not supported by TensorFlow. Use Python 3.10-3.12.",
                    probe.version
                )
            } else if !missing.is_empty() {
                format!(
                    "Missing packages: {}. Install python/requirements.txt.",
                    missing.join(", ")
                )
            } else {
                "ML environment is ready.".into()
            };
            EnvironmentStatus {
                ready,
                python: python.to_string_lossy().to_string(),
                executable: probe.executable,
                version: probe.version,
                supported_python: probe.supported_python,
                packages: probe.packages,
                message,
                script_dir: scripts.to_string_lossy().to_string(),
            }
        }
        _ => EnvironmentStatus {
            ready: false,
            python: python.to_string_lossy().to_string(),
            executable: String::new(),
            version: String::new(),
            supported_python: false,
            packages: BTreeMap::new(),
            message: format!(
                "Python environment probe failed. {}",
                stderr.lines().last().unwrap_or("").trim()
            )
            .trim()
            .to_string(),
            script_dir: scripts.to_string_lossy().to_string(),
        },
    }
}

pub fn start_job(
    app: AppHandle,
    jobs: &JobState,
    project: &str,
    kind: &str,
    script_name: &str,
    args: Vec<String>,
    envs: Vec<(String, String)>,
) -> Result<JobStarted, String> {
    let mut slot = jobs
        .current
        .lock()
        .map_err(|_| "Job manager is unavailable.".to_string())?;
    if let Some(current) = slot.as_ref() {
        return Err(format!(
            "A {} job is already running ({}). Wait for it to finish or cancel it.",
            current.kind, current.id
        ));
    }

    let python = find_python(&app)?;
    let scripts = python_scripts_dir(&app)?;
    let script = scripts.join(script_name);
    if !script.is_file() {
        return Err(format!("Pipeline script not found: {}", script.display()));
    }

    let mut command = command_for(&python);
    command
        .arg(&script)
        .args(&args)
        .current_dir(&scripts)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("TF_CPP_MIN_LOG_LEVEL", "2")
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Cannot launch Python ({}): {error}", python.display()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Cannot capture Python stdout.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Cannot capture Python stderr.".to_string())?;

    let job_id = Uuid::new_v4().to_string();
    let child = Arc::new(Mutex::new(Some(child)));
    let canceled = Arc::new(AtomicBool::new(false));
    let result: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));

    *slot = Some(ActiveJob {
        id: job_id.clone(),
        project: project.to_string(),
        kind: kind.to_string(),
        child: Arc::clone(&child),
        canceled: Arc::clone(&canceled),
    });
    drop(slot);

    let stdout_app = app.clone();
    let stdout_job = JobTag::new(&job_id, project, kind);
    let stdout_result = Arc::clone(&result);
    let stdout_reader = thread::Builder::new()
        .name("wakeforge-python-stdout".into())
        .spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                forward_line(
                    &stdout_app,
                    &stdout_job,
                    "stdout",
                    &line,
                    Some(&stdout_result),
                );
            }
        })
        .map_err(|error| format!("Cannot spawn Python stdout reader: {error}"))?;

    let stderr_app = app.clone();
    let stderr_job = JobTag::new(&job_id, project, kind);
    let stderr_reader = thread::Builder::new()
        .name("wakeforge-python-stderr".into())
        .spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                forward_line(&stderr_app, &stderr_job, "stderr", &line, None);
            }
        })
        .map_err(|error| format!("Cannot spawn Python stderr reader: {error}"))?;

    let wait_app = app.clone();
    let wait_job = JobTag::new(&job_id, project, kind);
    thread::Builder::new()
        .name("wakeforge-python-waiter".into())
        .spawn(move || {
            let (exit_code, wait_error) = loop {
                let mut guard = match child.lock() {
                    Ok(guard) => guard,
                    Err(_) => break (None, Some("Python process lock poisoned.".to_string())),
                };
                let status = match guard.as_mut() {
                    Some(process) => match process.try_wait() {
                        Ok(Some(status)) => Some(status),
                        Ok(None) => None,
                        Err(error) => {
                            let _ = process.kill();
                            break (None, Some(format!("Cannot wait for Python: {error}")));
                        }
                    },
                    None => {
                        break (
                            None,
                            Some("Python process handle disappeared.".to_string()),
                        )
                    }
                };
                drop(guard);
                if let Some(status) = status {
                    break (status.code(), None);
                }
                thread::sleep(Duration::from_millis(120));
            };

            // Ensure every log/progress line has been emitted before the final
            // event reaches the UI.
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();

            let canceled_flag = canceled.load(Ordering::SeqCst);
            let payload = result
                .lock()
                .ok()
                .and_then(|mut value| value.take());
            let success = exit_code == Some(0) && !canceled_flag;
            let message = if canceled_flag {
                "Job canceled.".to_string()
            } else if let Some(error) = wait_error {
                error
            } else if success {
                "Job completed.".to_string()
            } else {
                format!(
                    "Python exited with code {}.",
                    exit_code
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "unknown".into())
                )
            };

            // Clear the active slot before notifying listeners so the UI may
            // immediately start a new job from the finished handler.
            if let Some(state) = wait_app.try_state::<crate::AppState>() {
                if let Ok(mut slot) = state.jobs.current.lock() {
                    if slot.as_ref().map(|job| job.id.as_str()) == Some(wait_job.id.as_str()) {
                        *slot = None;
                    }
                }
            }

            let event = JobFinishedEvent {
                job_id: wait_job.id,
                project: wait_job.project,
                kind: wait_job.kind,
                success,
                canceled: canceled_flag,
                exit_code,
                result: payload,
                message,
            };
            let _ = wait_app.emit("job-finished", event);
        })
        .map_err(|error| format!("Cannot spawn Python waiter: {error}"))?;

    Ok(JobStarted {
        job_id,
        project: project.to_string(),
        kind: kind.to_string(),
        python: python.to_string_lossy().to_string(),
        script: script.to_string_lossy().to_string(),
    })
}

/// Identifies a job in every event so the UI can ignore events that do not
/// belong to the project it is showing.
struct JobTag {
    id: String,
    project: String,
    kind: String,
}

impl JobTag {
    fn new(id: &str, project: &str, kind: &str) -> Self {
        Self {
            id: id.to_string(),
            project: project.to_string(),
            kind: kind.to_string(),
        }
    }
}

fn forward_line(
    app: &AppHandle,
    job: &JobTag,
    stream: &str,
    line: &str,
    result_sink: Option<&Arc<Mutex<Option<Value>>>>,
) {
    let trimmed = line.trim();
    if let Some(payload) = trimmed.strip_prefix(PROGRESS_PREFIX) {
        if let Ok(Value::Object(mut object)) = serde_json::from_str::<Value>(payload.trim()) {
            object.insert("jobId".into(), Value::String(job.id.clone()));
            object.insert("project".into(), Value::String(job.project.clone()));
            object.insert("kind".into(), Value::String(job.kind.clone()));
            let _ = app.emit("job-progress", Value::Object(object));
        }
        return;
    }
    if let Some(payload) = trimmed.strip_prefix(RESULT_PREFIX) {
        if let Ok(value) = serde_json::from_str::<Value>(payload.trim()) {
            if let Some(sink) = result_sink {
                if let Ok(mut slot) = sink.lock() {
                    *slot = Some(value);
                }
            }
        }
        return;
    }
    if trimmed.is_empty() {
        return;
    }

    let event = JobLogEvent {
        job_id: job.id.clone(),
        project: job.project.clone(),
        kind: job.kind.clone(),
        stream: stream.to_string(),
        line: line.to_string(),
        timestamp: Utc::now().to_rfc3339(),
    };
    let _ = app.emit("job-log", event);
}

/// Run a short script to completion and return its `@@RESULT@@` payload.
/// Used for quick queries (no events, no job slot).
pub fn run_script(app: &AppHandle, script_name: &str, args: &[String]) -> Result<Value, String> {
    let python = find_python(app)?;
    let scripts = python_scripts_dir(app)?;
    let output = command_for(&python)
        .arg(scripts.join(script_name))
        .args(args)
        .current_dir(&scripts)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("Cannot launch Python ({}): {error}", python.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(value) = stdout
        .lines()
        .rev()
        .find_map(|line| line.trim().strip_prefix(RESULT_PREFIX))
        .and_then(|payload| serde_json::from_str::<Value>(payload.trim()).ok())
    {
        return Ok(value);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let reason = stderr
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("no output")
        .trim()
        .to_string();
    Err(format!("{script_name} failed: {reason}"))
}

pub fn cancel_job(app: &AppHandle) -> Result<(), String> {
    let app_state = app
        .try_state::<crate::AppState>()
        .ok_or_else(|| "Application state is unavailable.".to_string())?;
    let mut slot = app_state
        .jobs
        .current
        .lock()
        .map_err(|_| "Job manager is unavailable.".to_string())?;
    let Some(job) = slot.as_mut() else {
        return Err("No job is running.".into());
    };
    job.canceled.store(true, Ordering::SeqCst);
    let mut child_guard = job
        .child
        .lock()
        .map_err(|_| "Python process is unavailable.".to_string())?;
    if let Some(child) = child_guard.as_mut() {
        child
            .kill()
            .map_err(|error| format!("Cannot cancel the Python process: {error}"))?;
    }
    Ok(())
}
