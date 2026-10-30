import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { errorMessage } from "@/lib/api";
import { guessLanguage, voiceApi, type ProjectVoice } from "@/lib/voice";
import type { JobFinishedEvent, ProjectSummary } from "@/types";

const SAVE_DELAY_MS = 400;

type ProjectVoiceContextValue = {
  voice: ProjectVoice;
  /** The phrase spoken to the device; the project name until the user changes it. */
  spokenText: string;
  language: string;
  update: (change: (current: ProjectVoice) => ProjectVoice) => void;
  /** Persist pending edits now; jobs read voice.json from disk. */
  flush: () => Promise<void>;
  saveError: string | null;
};

const ProjectVoiceContext = createContext<ProjectVoiceContextValue | null>(null);

function withDefaults(summary: ProjectSummary): ProjectVoice {
  const voice = summary.voice ?? {};
  const spokenText = voice.spokenText ?? summary.phrase;
  return {
    ...voice,
    spokenText,
    language: voice.language || guessLanguage(spokenText),
    groups: voice.groups ?? {},
    synthesis: voice.synthesis ?? { voicesPerNegative: 2, includePositives: true },
  };
}

/**
 * One editable copy of the project's voice.json (spoken phrase, language, word
 * lists) shared by every panel, saved with a short debounce. Mount it inside
 * a container keyed by the project so each project starts fresh.
 */
export function ProjectVoiceProvider({
  summary,
  jobFinished,
  children,
}: {
  summary: ProjectSummary;
  jobFinished: JobFinishedEvent | null;
  children: ReactNode;
}) {
  const project = summary.slug;
  const [voice, setVoice] = useState<ProjectVoice>(() => withDefaults(summary));
  const [saveError, setSaveError] = useState<string | null>(null);
  const saveTimer = useRef<number | null>(null);
  const pendingSave = useRef<ProjectVoice | null>(null);
  const handledJob = useRef<string | null>(null);
  const voiceRef = useRef(voice);
  voiceRef.current = voice;

  const save = useCallback(async () => {
    if (saveTimer.current !== null) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const data = pendingSave.current;
    if (!data) return;
    pendingSave.current = null;
    try {
      await voiceApi.saveProject(project, data);
      setSaveError(null);
    } catch (error) {
      setSaveError(errorMessage(error));
    }
  }, [project]);

  const update = useCallback(
    (change: (current: ProjectVoice) => ProjectVoice) => {
      setVoice((current) => {
        const next = change(current);
        pendingSave.current = next;
        if (saveTimer.current !== null) window.clearTimeout(saveTimer.current);
        saveTimer.current = window.setTimeout(() => void save(), SAVE_DELAY_MS);
        return next;
      });
    },
    [save],
  );

  const flush = useCallback(async () => {
    // Also persist defaults (phrase, guessed language) the user never touched.
    pendingSave.current = voiceRef.current;
    await save();
  }, [save]);

  useEffect(() => () => void save(), [save]);

  // Apply results of this project's finished jobs exactly once.
  useEffect(() => {
    if (!jobFinished || !jobFinished.success || jobFinished.project !== project) return;
    if (handledJob.current === jobFinished.jobId) return;
    handledJob.current = jobFinished.jobId;
    const result = (jobFinished.result ?? {}) as Record<string, unknown>;
    const now = new Date().toISOString();
    if (jobFinished.kind === "transcribe") {
      update((current) => ({
        ...current,
        transcription: { ...(result as unknown as NonNullable<ProjectVoice["transcription"]>), at: now },
      }));
    } else if (jobFinished.kind === "words") {
      const generated = (result.groups ?? {}) as Record<string, { items: string[] }>;
      const provider = typeof result.provider === "string" ? result.provider : undefined;
      update((current) => {
        const nextGroups = { ...(current.groups ?? {}) };
        for (const [id, value] of Object.entries(generated)) {
          const existing = nextGroups[id]?.items ?? [];
          nextGroups[id] = { items: [...existing, ...(value.items ?? [])], generatedAt: now, provider };
        }
        return { ...current, groups: nextGroups };
      });
    }
  }, [jobFinished, project, update]);

  const spokenText = voice.spokenText?.trim() || summary.phrase;
  const value: ProjectVoiceContextValue = {
    voice,
    spokenText,
    language: voice.language || guessLanguage(spokenText),
    update,
    flush,
    saveError,
  };
  return <ProjectVoiceContext.Provider value={value}>{children}</ProjectVoiceContext.Provider>;
}

export function useProjectVoice(): ProjectVoiceContextValue {
  const value = useContext(ProjectVoiceContext);
  if (!value) throw new Error("useProjectVoice must be used inside ProjectVoiceProvider");
  return value;
}
