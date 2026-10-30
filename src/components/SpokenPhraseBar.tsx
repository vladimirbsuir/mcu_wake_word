import { AudioLines, Check, Ear, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useProjectVoice } from "@/hooks/useProjectVoice";
import type { PipelineKind } from "@/hooks/useJobRunner";
import {
  categoryReadiness,
  isUsable,
  voiceApi,
  type VoiceCategory,
  type VoiceSettings,
  type VoiceStatus,
} from "@/lib/voice";
import type { JobStarted, ProjectSummary } from "@/types";

type SpokenPhraseBarProps = {
  summary: ProjectSummary;
  status: VoiceStatus | null;
  settings: VoiceSettings | null;
  jobBusy: boolean;
  jobKind: PipelineKind;
  jobProject: string | null;
  interactionDisabled: boolean;
  onRunJob: (kind: Exclude<PipelineKind, null>, starter: () => Promise<JobStarted>) => void;
  onOpenModels: (category: VoiceCategory) => void;
};

/**
 * The phrase spoken to the device and its language, shared by every step:
 * recorder prompts, word lists, synthetic voices, dataset and model metadata,
 * and the streaming test. The project name can be anything.
 */
export function SpokenPhraseBar({
  summary,
  status,
  settings,
  jobBusy,
  jobKind,
  jobProject,
  interactionDisabled,
  onRunJob,
  onOpenModels,
}: SpokenPhraseBarProps) {
  const { voice, language, update, flush, saveError } = useProjectVoice();
  const stt = categoryReadiness("stt", language, settings, status);
  const disabled = jobBusy || interactionDisabled;
  const transcribing = jobBusy && jobKind === "transcribe" && jobProject === summary.slug;
  const transcription = voice.transcription;
  const suggestionDiffers =
    transcription?.suggestion &&
    transcription.suggestion.toLowerCase() !== (voice.spokenText ?? "").trim().toLowerCase();

  const transcribe = async () => {
    await flush();
    onRunJob("transcribe", () => voiceApi.transcribe(summary.slug));
  };

  return (
    <section className="panel flex flex-wrap items-end gap-x-5 gap-y-3 px-5 py-4">
      <div className="min-w-[220px] flex-1 space-y-1.5">
        <Label htmlFor="spoken-text">Wake phrase · what you say to the device</Label>
        <Input
          id="spoken-text"
          value={voice.spokenText ?? ""}
          onChange={(event) => update((current) => ({ ...current, spokenText: event.target.value }))}
          placeholder={summary.phrase}
          disabled={disabled}
          maxLength={80}
          className="h-9 font-medium"
        />
      </div>
      <div className="w-[170px] space-y-1.5">
        <Label htmlFor="phrase-language">Language</Label>
        <select
          id="phrase-language"
          value={language}
          onChange={(event) => update((current) => ({ ...current, language: event.target.value }))}
          disabled={disabled}
          className="focus-ring flex h-9 w-full rounded-md border border-input bg-background/60 px-2 text-xs shadow-sm disabled:opacity-50"
        >
          {(status?.languages ?? []).map((item) => (
            <option key={item.code} value={item.code}>
              {item.native} ({item.code})
            </option>
          ))}
          {!status ? <option value={language}>{language}</option> : null}
        </select>
      </div>
      <div className="flex min-w-[240px] flex-1 flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-9"
            disabled={disabled || !summary.recordings.length || !isUsable(stt)}
            title={!isUsable(stt) ? "Set up speech-to-text first" : "Check that every take says this phrase"}
            onClick={() => void transcribe()}
          >
            {transcribing ? <Loader2 className="animate-spin" /> : <Ear className="h-3 w-3" />}
            Check {summary.recordings.length} takes
          </Button>
          {!isUsable(stt) ? (
            <button
              type="button"
              onClick={() => onOpenModels("stt")}
              className="text-[10.5px] text-zinc-600 underline-offset-2 hover:text-zinc-300 hover:underline"
            >
              set up speech-to-text
            </button>
          ) : null}
        </div>
        {transcription ? (
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <AudioLines className="h-3 w-3 text-zinc-600" />
            <span className="text-zinc-500">Heard</span>
            <span className="font-medium text-zinc-100">“{transcription.suggestion}”</span>
            <Badge variant="outline">{Math.round(transcription.agreement * 100)}% agree</Badge>
            {suggestionDiffers ? (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[10px]"
                onClick={() => update((current) => ({ ...current, spokenText: transcription.suggestion }))}
                disabled={disabled}
              >
                <Check className="h-3 w-3" /> Use
              </Button>
            ) : null}
            {transcription.agreement < 1 ? (
              <details className="w-full text-zinc-500">
                <summary className="cursor-pointer text-[10.5px]">Takes that disagree</summary>
                <ul className="mt-1 space-y-0.5 font-mono text-[10px]">
                  {transcription.transcripts
                    .filter((item) => item.text.toLowerCase() !== transcription.suggestion.toLowerCase())
                    .map((item) => (
                      <li key={item.file}>
                        {item.file.replace(/\.wav$/, "")}: {item.text || "—"}
                      </li>
                    ))}
                </ul>
              </details>
            ) : null}
          </div>
        ) : null}
        {saveError ? <p className="text-[10.5px] text-red-300">{saveError}</p> : null}
      </div>
    </section>
  );
}
