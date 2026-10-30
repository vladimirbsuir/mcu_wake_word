import { invoke } from "@tauri-apps/api/core";
import type { JobStarted } from "@/types";

export type VoiceCategory = "stt" | "llm" | "tts";

export const VOICE_CATEGORIES: Array<{ id: VoiceCategory; title: string; short: string; purpose: string }> = [
  {
    id: "stt",
    title: "Speech-to-text",
    short: "STT",
    purpose: "Reads the spoken phrase back from your takes.",
  },
  {
    id: "llm",
    title: "Word lists (LLM)",
    short: "LLM",
    purpose: "Writes similar-sounding words and everyday phrases the detector must ignore.",
  },
  {
    id: "tts",
    title: "Text-to-speech",
    short: "TTS",
    purpose: "Voices those lists, and your wake phrase, in many synthetic voices.",
  },
];

export type CatalogModel = {
  id: string;
  category: VoiceCategory | "runtime";
  engine: string;
  title: string;
  description: string;
  /** null means effectively any language. */
  languages: string[] | null;
  recommendedFor: string[];
  sizeMb: number;
  ramGb: number;
  license: string;
  voices: number;
  speakers: number;
  dependsOn: string[];
  installed: boolean;
  requirements: Array<{ package: string; module: string; installed: boolean; sizeMb: number }>;
};

export type WordGroupInfo = {
  id: string;
  title: string;
  description: string;
  label: "negative" | "positive";
  defaultCount: number;
};

export type LanguageInfo = { code: string; name: string; native: string };

export type VoiceStatus = {
  models: CatalogModel[];
  profileTypes: Record<string, { title: string; description: string; categories: VoiceCategory[] }>;
  groups: WordGroupInfo[];
  languages: LanguageInfo[];
  corpora: SpeechCorpus[];
  python: { version: string; executable: string };
  modelsDir: string;
};

/** Downloadable speech: "negatives" feed the dataset, "eval" feeds the streaming test. */
export type SpeechCorpus = {
  id: string;
  role: "negatives" | "eval";
  title: string;
  description: string;
  language: string;
  hours: number;
  sizeMb: number;
  license: string;
  homepage: string;
  installed: boolean;
  utterances: number | null;
  installedHours: number | null;
};

/** Tabs of the AI models dialog: the three provider categories plus speech data. */
export type ModelsTab = VoiceCategory | "data";

export type VoiceChoice = { provider: string; model?: string };

export type VoiceProfile = {
  id: string;
  type: "openai";
  name: string;
  baseUrl: string;
  models: Record<VoiceCategory, string>;
  voices: string[];
  languages: string[];
  hasKey: boolean;
};

export type VoiceSettings = {
  active: Partial<Record<VoiceCategory, VoiceChoice | null>>;
  profiles: VoiceProfile[];
};

export type WordGroupData = { items: string[]; generatedAt?: string; provider?: string };

export type TakeTranscript = { file: string; text: string };

export type ProjectVoice = {
  spokenText?: string;
  language?: string;
  groups?: Record<string, WordGroupData>;
  transcription?: { suggestion: string; agreement: number; transcripts: TakeTranscript[]; at: string };
  synthesis?: {
    voicesPerNegative: number;
    includePositives: boolean;
    /** TTS engines to voice with; the active TTS when unset. */
    engines?: VoiceChoice[];
  };
};

export type SyntheticClip = {
  file: string;
  label: "positive" | "negative";
  group: string;
  text: string;
  voice: string;
  voiceName: string;
  speed: number;
  seconds: number;
};

export type SyntheticSummary = {
  createdAt: string;
  engine: string;
  language: string;
  spokenText: string;
  voices: number;
  counts: { positive: number; negative: number };
  failures: number;
  engines?: Array<{ key: string; label: string; voices: number; wanted: number; synthesized: number; reused: number; failures: number }>;
  samples: Partial<Record<"positive" | "negative", SyntheticClip[]>>;
};

export const LOCAL_PROVIDER = "local";

export const voiceApi = {
  status() {
    return invoke<VoiceStatus>("voice_status");
  },
  settings() {
    return invoke<VoiceSettings>("voice_settings_get");
  },
  saveSettings(settings: VoiceSettings) {
    return invoke<VoiceSettings>("voice_settings_save", { settings });
  },
  setApiKey(profileId: string, key: string | null) {
    return invoke<VoiceSettings>("voice_set_api_key", { profileId, key });
  },
  installModel(model: string) {
    return invoke<JobStarted>("voice_install_model", { model });
  },
  installPackages(model: string) {
    return invoke<JobStarted>("voice_install_packages", { model });
  },
  removeModel(model: string) {
    return invoke<void>("voice_remove_model", { model });
  },
  installCorpus(corpus: string) {
    return invoke<JobStarted>("voice_install_corpus", { corpus });
  },
  removeCorpus(corpus: string) {
    return invoke<void>("voice_remove_corpus", { corpus });
  },
  check(category: VoiceCategory, language: string | null) {
    return invoke<JobStarted>("voice_check", { category, language });
  },
  saveProject(project: string, data: ProjectVoice) {
    return invoke<void>("voice_project_save", { project, data });
  },
  transcribe(project: string) {
    return invoke<JobStarted>("voice_transcribe", { project });
  },
  generateWords(project: string, groups: Record<string, number>) {
    return invoke<JobStarted>("voice_generate_words", { project, groups });
  },
  synthesize(project: string, includePositives: boolean, voicesPerNegative: number) {
    return invoke<JobStarted>("voice_synthesize", { project, includePositives, voicesPerNegative });
  },
  readSyntheticAudio(project: string, file: string) {
    return invoke<number[]>("read_synthetic_audio", { project, file });
  },
  deleteSynthetic(project: string) {
    return invoke<void>("delete_synthetic", { project });
  },
};

export type ReadinessState = "ready" | "unverified" | "download" | "packages" | "unsupported" | "unselected";

export type Readiness = {
  state: ReadinessState;
  /** Short label for chips. */
  label: string;
  /** What is selected, if anything. */
  provider: string | null;
};

export function profileServes(profile: VoiceProfile, category: VoiceCategory): boolean {
  return Boolean(profile.models?.[category]?.trim());
}

export function supportsLanguage(languages: string[] | null | undefined, language: string | null): boolean {
  if (!language || !languages || languages.length === 0) return true;
  return languages.includes(language);
}

export function modelReadiness(model: CatalogModel, status: VoiceStatus): ReadinessState {
  const dependencies = model.dependsOn.map((id) => status.models.find((item) => item.id === id));
  if (!model.installed || dependencies.some((item) => !item?.installed)) return "download";
  if (model.requirements.some((requirement) => !requirement.installed)) return "packages";
  return "ready";
}

/** Whether the active choice of a category can run for the given language. */
export function categoryReadiness(
  category: VoiceCategory,
  language: string | null,
  settings: VoiceSettings | null,
  status: VoiceStatus | null,
): Readiness {
  return choiceReadiness(category, settings?.active?.[category] ?? null, language, settings, status);
}

/** Whether one provider choice can run for the given language. */
export function choiceReadiness(
  category: VoiceCategory,
  choice: VoiceChoice | null,
  language: string | null,
  settings: VoiceSettings | null,
  status: VoiceStatus | null,
): Readiness {
  if (!choice?.provider || !status) return { state: "unselected", label: "not selected", provider: null };
  if (choice.provider === LOCAL_PROVIDER) {
    const model = status.models.find((item) => item.id === choice.model);
    if (!model) return { state: "unselected", label: "not selected", provider: null };
    if (!supportsLanguage(model.languages, language)) {
      return { state: "unsupported", label: "language not supported", provider: model.title };
    }
    const state = modelReadiness(model, status);
    const label = state === "ready" ? "ready" : state === "download" ? "not downloaded" : "packages missing";
    return { state, label, provider: model.title };
  }
  const profile = settings?.profiles.find((item) => item.id === choice.provider);
  if (!profile || !profileServes(profile, category)) {
    return { state: "unselected", label: "not selected", provider: null };
  }
  const name = `${profile.name} · ${profile.models[category]}`;
  if (profile.languages.length && language && !profile.languages.includes(language)) {
    return { state: "unsupported", label: "language not supported", provider: name };
  }
  return {
    state: profile.languages.length ? "ready" : "unverified",
    label: profile.languages.length ? "ready" : "language not declared",
    provider: name,
  };
}

export function sameChoice(a: VoiceChoice, b: VoiceChoice): boolean {
  return a.provider === b.provider && (a.provider !== LOCAL_PROVIDER || a.model === b.model);
}

/** Every TTS that could voice clips: installed local models and endpoints with a TTS model. */
export function ttsCandidates(settings: VoiceSettings | null, status: VoiceStatus | null): VoiceChoice[] {
  const local = (status?.models ?? [])
    .filter((model) => model.category === "tts")
    .map((model) => ({ provider: LOCAL_PROVIDER, model: model.id }));
  const remote = (settings?.profiles ?? [])
    .filter((profile) => profileServes(profile, "tts"))
    .map((profile) => ({ provider: profile.id }));
  return [...local, ...remote];
}

/** How many voices a TTS choice offers, when known. */
export function ttsVoiceCount(choice: VoiceChoice, settings: VoiceSettings | null, status: VoiceStatus | null): number {
  if (choice.provider === LOCAL_PROVIDER) {
    const model = status?.models.find((item) => item.id === choice.model);
    return model ? model.speakers || model.voices || 1 : 1;
  }
  const profile = settings?.profiles.find((item) => item.id === choice.provider);
  return profile?.voices.length || 9;
}

export function isUsable(readiness: Readiness): boolean {
  return readiness.state === "ready" || readiness.state === "unverified";
}

/** A reasonable starting language from the phrase's script. */
export function guessLanguage(text: string): string {
  if (/[Ѐ-ӿ]/.test(text)) return /[іїєґ]/i.test(text) ? "uk" : "ru";
  if (/[Ͱ-Ͽ]/.test(text)) return "el";
  if (/[؀-ۿ]/.test(text)) return "ar";
  if (/[가-힯]/.test(text)) return "ko";
  if (/[぀-ヿ]/.test(text)) return "ja";
  if (/[一-鿿]/.test(text)) return "zh";
  if (/[äöüß]/i.test(text)) return "de";
  return "en";
}

export function newProfileId(): string {
  return `p_${Math.random().toString(36).slice(2, 10)}`;
}

export function playWav(bytes: number[]): Promise<void> {
  const url = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: "audio/wav" }));
  const audio = new Audio(url);
  return new Promise((resolve) => {
    const done = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onended = done;
    audio.onerror = done;
    void audio.play().catch(done);
  });
}
