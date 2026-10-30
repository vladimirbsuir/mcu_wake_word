//! AI provider settings, API keys and project voice data.
//!
//! The Python side (`python/voice.py`, `python/voicekit/`) owns the model
//! catalog and every engine. This module only stores the user's choices,
//! keeps API keys in the OS credential store and prepares job arguments.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Manager};

const SETTINGS_FILE: &str = "voice_settings.json";
const PROJECT_VOICE_FILE: &str = "voice.json";
const SYNTHETIC_DIR: &str = "synthetic";
const KEYRING_SERVICE: &str = "WakeForge Studio";
pub const CATEGORIES: [&str; 3] = ["stt", "llm", "tts"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Choice {
    /// "local" for a catalog model, otherwise a profile id.
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActiveChoices {
    #[serde(default)]
    pub stt: Option<Choice>,
    #[serde(default)]
    pub llm: Option<Choice>,
    #[serde(default)]
    pub tts: Option<Choice>,
}

impl ActiveChoices {
    fn all(&self) -> [&Option<Choice>; 3] {
        [&self.stt, &self.llm, &self.tts]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileModels {
    #[serde(default)]
    pub llm: String,
    #[serde(default)]
    pub tts: String,
    #[serde(default)]
    pub stt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub models: ProfileModels,
    /// TTS voice names to use; empty means the provider defaults.
    #[serde(default)]
    pub voices: Vec<String>,
    /// Declared supported languages; empty means "any / not declared".
    #[serde(default)]
    pub languages: Vec<String>,
    /// Filled in on read from the credential store; never written to disk.
    #[serde(default, skip_deserializing)]
    pub has_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSettings {
    #[serde(default)]
    pub active: ActiveChoices,
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

fn app_data(app: &AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Cannot resolve app data directory: {error}"))?;
    fs::create_dir_all(&root).map_err(|error| format!("Cannot create app data directory: {error}"))?;
    Ok(root)
}

pub fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app_data(app)?.join("models");
    fs::create_dir_all(&dir).map_err(|error| format!("Cannot create models directory: {error}"))?;
    Ok(dir)
}

/// Prepared speech corpora (see python/voicekit/corpora.py), shared by all projects.
pub fn datasets_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app_data(app)?.join("datasets");
    fs::create_dir_all(&dir).map_err(|error| format!("Cannot create datasets directory: {error}"))?;
    Ok(dir)
}

/// Folders of installed corpora with the given role ("negatives" or "eval").
pub fn installed_corpora(app: &AppHandle, role: &str) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let entries = match fs::read_dir(datasets_dir(app)?) {
        Ok(entries) => entries,
        Err(_) => return Ok(found),
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let marker = fs::read_to_string(dir.join(".installed.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        if marker
            .as_ref()
            .and_then(|value| value.get("role"))
            .and_then(Value::as_str)
            == Some(role)
            && dir.join("index.json").is_file()
        {
            found.push(dir);
        }
    }
    found.sort();
    Ok(found)
}

/// The phrase spoken to the device (voice.json), or the project name when unset.
pub fn spoken_phrase(project_dir: &Path, fallback: &str) -> String {
    read_project_voice(project_dir)
        .get("spokenText")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

pub fn remove_corpus(app: &AppHandle, corpus: &str) -> Result<(), String> {
    if !valid_id(corpus) {
        return Err("Invalid corpus id.".into());
    }
    let dir = datasets_dir(app)?.join(corpus);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|error| format!("Cannot delete the corpus files: {error}"))?;
    }
    Ok(())
}

pub fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data(app)?.join(SETTINGS_FILE))
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn keyring_entry(profile_id: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYRING_SERVICE, &format!("api-key:{profile_id}"))
        .map_err(|error| format!("Credential store is unavailable: {error}"))
}

fn read_key(profile_id: &str) -> Option<String> {
    keyring_entry(profile_id)
        .ok()?
        .get_password()
        .ok()
        .filter(|key| !key.is_empty())
}

pub fn load_settings(app: &AppHandle) -> Result<VoiceSettings, String> {
    let path = settings_path(app)?;
    let mut settings = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<VoiceSettings>(&text)
            .map_err(|error| format!("AI settings file is corrupted ({}): {error}", path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => VoiceSettings::default(),
        Err(error) => return Err(format!("Cannot read AI settings: {error}")),
    };
    for profile in &mut settings.profiles {
        profile.has_key = read_key(&profile.id).is_some();
    }
    Ok(settings)
}

pub fn save_settings(app: &AppHandle, mut settings: VoiceSettings) -> Result<VoiceSettings, String> {
    let mut seen = std::collections::HashSet::new();
    for profile in &mut settings.profiles {
        if !valid_id(&profile.id) || !seen.insert(profile.id.clone()) {
            return Err(format!("Invalid or duplicate endpoint id '{}'.", profile.id));
        }
        if profile.kind != "openai" {
            return Err(format!("Unknown endpoint type '{}'.", profile.kind));
        }
        profile.name = profile.name.trim().to_string();
        if profile.name.is_empty() {
            return Err("Every endpoint needs a name.".into());
        }
        profile.base_url = profile.base_url.trim().trim_end_matches('/').to_string();
        if !(profile.base_url.starts_with("http://") || profile.base_url.starts_with("https://")) {
            return Err(format!("Endpoint '{}' needs an http(s) base URL.", profile.name));
        }
    }
    // A choice that points to a removed endpoint is dropped, not kept dangling.
    for choice in [&mut settings.active.stt, &mut settings.active.llm, &mut settings.active.tts] {
        let dangling = choice.as_ref().is_some_and(|value| {
            value.provider != "local" && !seen.contains(&value.provider)
        });
        if dangling {
            *choice = None;
        }
    }
    let payload = serde_json::to_string_pretty(&settings)
        .map_err(|error| format!("Cannot serialize AI settings: {error}"))?;
    fs::write(settings_path(app)?, payload).map_err(|error| format!("Cannot save AI settings: {error}"))?;
    load_settings(app)
}

pub fn set_api_key(profile_id: &str, key: Option<&str>) -> Result<(), String> {
    if !valid_id(profile_id) {
        return Err("Invalid endpoint id.".into());
    }
    let entry = keyring_entry(profile_id)?;
    match key.map(str::trim).filter(|key| !key.is_empty()) {
        Some(key) => entry
            .set_password(key)
            .map_err(|error| format!("Cannot store the API key: {error}")),
        None => match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(format!("Cannot remove the API key: {error}")),
        },
    }
}

/// Environment variables with the API keys of the endpoints currently in use.
/// Keys travel to the Python job only through its environment.
pub fn job_env(app: &AppHandle) -> Result<Vec<(String, String)>, String> {
    let settings = load_settings(app)?;
    let mut envs = Vec::new();
    for choice in settings.active.all().into_iter().flatten() {
        if choice.provider == "local" {
            continue;
        }
        if let Some(key) = read_key(&choice.provider) {
            let name: String = choice
                .provider
                .to_ascii_uppercase()
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            envs.push((format!("WAKEFORGE_API_KEY_{name}"), key));
        }
    }
    Ok(envs)
}

pub fn valid_model_id(model: &str) -> bool {
    valid_id(model)
}

pub fn remove_model(app: &AppHandle, model: &str) -> Result<(), String> {
    if !valid_model_id(model) {
        return Err("Invalid model id.".into());
    }
    let dir = models_dir(app)?.join(model);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|error| format!("Cannot delete the model files: {error}"))?;
    }
    Ok(())
}

// ------------------------------------------------------------ project data

pub fn voice_file(project_dir: &Path) -> PathBuf {
    project_dir.join(PROJECT_VOICE_FILE)
}

pub fn synthetic_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(SYNTHETIC_DIR)
}

/// The project's spoken phrase, language and word lists (free-form JSON owned
/// by the UI, validated loosely here).
pub fn read_project_voice(project_dir: &Path) -> Value {
    fs::read_to_string(voice_file(project_dir))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(Default::default()))
}

pub fn write_project_voice(project_dir: &Path, data: &Value) -> Result<(), String> {
    if !data.is_object() {
        return Err("Voice data must be a JSON object.".into());
    }
    let payload = serde_json::to_string_pretty(data)
        .map_err(|error| format!("Cannot serialize voice data: {error}"))?;
    fs::write(voice_file(project_dir), payload).map_err(|error| format!("Cannot save voice data: {error}"))
}

/// Synthetic set summary for the UI: the manifest without the full clip list,
/// plus a few example clips per label for playback.
pub fn synthetic_summary(project_dir: &Path) -> Option<Value> {
    let text = fs::read_to_string(synthetic_dir(project_dir).join("manifest.json")).ok()?;
    let mut manifest = serde_json::from_str::<Value>(&text).ok()?;
    let clips = manifest.get_mut("clips").map(Value::take).unwrap_or(Value::Null);
    let mut samples: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut seen_voices = std::collections::HashSet::new();
    for clip in clips.as_array().into_iter().flatten() {
        let label = clip.get("label").and_then(Value::as_str).unwrap_or("negative").to_string();
        let voice = clip.get("voice").and_then(Value::as_str).unwrap_or_default().to_string();
        let bucket = samples.entry(label.clone()).or_default();
        // One example per voice for positives, the first few for negatives.
        let take = if label == "positive" {
            seen_voices.insert(voice)
        } else {
            bucket.len() < 12
        };
        if take && bucket.len() < 40 {
            bucket.push(clip.clone());
        }
    }
    if let Some(object) = manifest.as_object_mut() {
        object.insert("samples".into(), serde_json::to_value(samples).unwrap_or(Value::Null));
    }
    Some(manifest)
}

pub fn read_synthetic_clip(project_dir: &Path, file: &str) -> Result<Vec<u8>, String> {
    let valid = !file.is_empty()
        && !file.contains("..")
        && file
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '.'))
        && file.ends_with(".wav");
    if !valid {
        return Err("Invalid clip path.".into());
    }
    fs::read(synthetic_dir(project_dir).join(file)).map_err(|error| format!("Cannot read clip: {error}"))
}

pub fn delete_synthetic(project_dir: &Path) -> Result<(), String> {
    let dir = synthetic_dir(project_dir);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|error| format!("Cannot delete synthetic clips: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_keys_round_trip_through_the_credential_store() {
        let id = "wakeforge-selftest";
        set_api_key(id, Some("sk-test-123")).unwrap();
        assert_eq!(read_key(id).as_deref(), Some("sk-test-123"));
        set_api_key(id, None).unwrap();
        assert_eq!(read_key(id), None);
        // Removing a missing key is not an error.
        set_api_key(id, None).unwrap();
    }

    #[test]
    fn profile_ids_are_restricted() {
        assert!(valid_id("p_ab12cd34"));
        assert!(!valid_id("../evil"));
        assert!(!valid_id(""));
    }
}
