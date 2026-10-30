import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AudioLines,
  BookOpenText,
  Loader2,
  Mic,
  Pause,
  Pencil,
  Play,
  SkipForward,
  Square,
  Trash2,
  Wand2,
  Waves,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Waveform } from "@/components/Waveform";
import { useProjectVoice } from "@/hooks/useProjectVoice";
import { useRecorder, AUTO_STOP_MS } from "@/hooks/useRecorder";
import { api, errorMessage } from "@/lib/api";
import { phraseSpecificCount, readingPrompts } from "@/lib/readingPrompts";
import { cn, formatBytes, formatDate, formatDuration } from "@/lib/utils";
import { voiceApi, type VoiceCategory } from "@/lib/voice";
import type { JobStarted, ProjectSummary, Recording, TakeKind } from "@/types";

type RecorderPanelProps = {
  summary: ProjectSummary;
  pipelineBusy: boolean;
  onRefresh: () => void | Promise<void>;
  onRecorderActivity?: (active: boolean) => void;
  /** A language model is set up, so prompts can include similar-sounding words. */
  llmReady: boolean;
  wordsRunning: boolean;
  onGenerateWords: (starter: () => Promise<JobStarted>) => void;
  defaultGroupCounts: Record<string, number>;
  onOpenModels: (category: VoiceCategory) => void;
};

function formatTimer(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function miniBars(seed: string): number[] {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = (hash * 31 + seed.charCodeAt(index)) >>> 0;
  }
  return Array.from({ length: 22 }, (_, index) => {
    hash = (hash * 1664525 + 1013904223) >>> 0;
    const envelope = 0.35 + 0.65 * Math.sin((index / 21) * Math.PI);
    return 0.16 + ((hash % 100) / 100) * 0.8 * envelope;
  });
}

export function RecorderPanel({
  summary,
  pipelineBusy,
  onRefresh,
  onRecorderActivity,
  llmReady,
  wordsRunning,
  onGenerateWords,
  defaultGroupCounts,
  onOpenModels,
}: RecorderPanelProps) {
  const project = summary.slug;
  const recorder = useRecorder(project, pipelineBusy, onRefresh);
  const [mode, setMode] = useState<TakeKind>("wake");
  const [promptSkip, setPromptSkip] = useState(0);
  const isWake = mode === "wake";
  const takes = isWake ? summary.recordings : summary.negativeTakes;
  const { voice, spokenText: spoken, language, flush } = useProjectVoice();
  const prompts = useMemo(
    () => readingPrompts(spoken, language, voice.groups),
    [spoken, language, voice.groups],
  );
  const specificPrompts = phraseSpecificCount(voice.groups);

  const generateWords = async () => {
    // The words job reads the phrase and language from voice.json.
    await flush();
    onGenerateWords(() => voiceApi.generateWords(project, defaultGroupCounts));
  };
  const prompt = prompts.length
    ? prompts[(summary.negativeTakes.length + promptSkip) % prompts.length]
    : null;
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [loadingPlaybackId, setLoadingPlaybackId] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const pipelineBusyRef = useRef(pipelineBusy);
  pipelineBusyRef.current = pipelineBusy;

  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.onended = null;
      audioRef.current = null;
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setPlayingId(null);
  }, []);

  useEffect(() => {
    onRecorderActivity?.(recorder.phase !== "idle");
  }, [onRecorderActivity, recorder.phase]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const handleRecordClick = async () => {
    setListError(null);
    if (recorder.isRecording) {
      stopPlayback();
      await recorder.stop();
    } else {
      stopPlayback();
      await recorder.start(mode, isWake ? null : prompt);
    }
  };

  const handlePlay = async (recording: Recording) => {
    if (playingId === recording.id) {
      stopPlayback();
      return;
    }
    if (loadingPlaybackId) return;
    stopPlayback();
    setLoadingPlaybackId(recording.id);
    setListError(null);
    try {
      const bytes = await api.readRecordingAudio(project, recording.id, mode);
      const blob = new Blob([new Uint8Array(bytes)], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        setPlayingId(null);
        audioRef.current = null;
        if (objectUrlRef.current === url) {
          URL.revokeObjectURL(url);
          objectUrlRef.current = null;
        }
      };
      audio.onerror = () => {
        setListError("Could not play this WAV file.");
        stopPlayback();
      };
      await audio.play();
      setPlayingId(recording.id);
    } catch (playError) {
      setListError(errorMessage(playError));
      stopPlayback();
    } finally {
      setLoadingPlaybackId(null);
    }
  };

  const handleDelete = async (recording: Recording) => {
    if (
      !window.confirm(
        `Delete this take?\n${formatDuration(recording.durationMs)} · ${formatDate(recording.createdAt)}`,
      )
    ) {
      return;
    }
    if (playingId === recording.id) stopPlayback();
    try {
      await api.deleteRecording(project, recording.id, mode);
      await onRefresh();
    } catch (deleteError) {
      setListError(errorMessage(deleteError));
    }
  };

  const handleRename = async (recording: Recording) => {
    const nextName = window.prompt("Rename this take", recording.id);
    if (!nextName || nextName.trim() === recording.id) return;
    if (playingId === recording.id) stopPlayback();
    try {
      await api.renameRecording(project, recording.id, nextName.trim());
      await onRefresh();
    } catch (renameError) {
      setListError(errorMessage(renameError));
    }
  };

  const showError = recorder.error || listError;
  const status =
    recorder.phase === "starting"
      ? "ARMING"
      : recorder.phase === "stopping"
        ? "PROCESSING"
        : recorder.phase === "recording"
          ? "RECORDING"
          : "READY";

  return (
    <section id="capture" className="panel relative scroll-mt-24 overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-56 bg-[radial-gradient(ellipse_at_top_right,hsl(78_82%_61%/0.055),transparent_60%)]" />
      <div className="relative p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">01</span>
              <span className="label-caps">Capture</span>
            </div>
            <h2 className="mt-2 text-lg font-semibold tracking-tight">
              {isWake ? "Record the wake phrase" : "Record other speech"}
            </h2>
            <p className="mt-1 max-w-xl text-xs leading-relaxed text-muted-foreground">
              {isWake
                ? `Say “${spoken}” the way you would in the final device. Silence is trimmed by an energy VAD and saved as 16 kHz mono PCM.`
                : "Read the prompt aloud in your normal voice. These takes teach the model that your voice saying anything else is not the wake phrase."}
            </p>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-2">
            <div className="flex rounded-md border border-border/80 bg-background/40 p-0.5">
              {(["wake", "negative"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => {
                    if (recorder.phase !== "idle") return;
                    stopPlayback();
                    setMode(value);
                  }}
                  disabled={recorder.phase !== "idle"}
                  className={cn(
                    "rounded px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors disabled:cursor-not-allowed",
                    mode === value
                      ? "bg-secondary text-zinc-100"
                      : "text-zinc-500 hover:text-zinc-300",
                  )}
                >
                  {value === "wake" ? "Wake phrase" : "Other speech"}
                </button>
              ))}
            </div>
            <Badge variant={takes.length ? "default" : "outline"}>
              {takes.length.toString().padStart(2, "0")} takes
            </Badge>
          </div>
        </div>

        {!isWake ? (
          <div className="mt-4 rounded-lg border border-cyan-300/15 bg-cyan-300/[0.03] px-4 py-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.16em] text-cyan-200/60">
                <BookOpenText className="h-3 w-3" />
                Read aloud
                {prompts.length ? (
                  <span className="text-zinc-600">
                    · {((summary.negativeTakes.length + promptSkip) % prompts.length) + 1}/
                    {prompts.length}
                  </span>
                ) : null}
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-[10px] text-zinc-500 hover:text-zinc-200"
                onClick={() => setPromptSkip((value) => value + 1)}
                disabled={!prompts.length || recorder.phase !== "idle"}
              >
                <SkipForward className="mr-1 h-3 w-3" />
                Next
              </Button>
            </div>
            <p className="mt-2 text-[15px] leading-snug text-zinc-100">
              {prompt ?? "Say anything you like: a request, a sentence, a few random words."}
            </p>
            {specificPrompts === 0 ? (
              <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-cyan-300/10 pt-2 text-[10.5px] text-zinc-500">
                <span>
                  Only general sentences so far. Words that sound like “{spoken}” make the strongest
                  negatives.
                </span>
                {llmReady ? (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => void generateWords()}
                    disabled={pipelineBusy || recorder.phase !== "idle"}
                  >
                    {wordsRunning ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wand2 className="h-3 w-3" />}
                    Generate similar words
                  </Button>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => onOpenModels("llm")}
                  >
                    Set up a language model
                  </Button>
                )}
              </div>
            ) : (
              <p className="mt-2 text-[10px] text-zinc-600">
                {specificPrompts} prompts come from the word lists for “{spoken}” (step 02).
              </p>
            )}
          </div>
        ) : null}

        <div className="relative mt-5 overflow-hidden rounded-lg border border-border/80 bg-[#080b09]">
          <div className="noise-bg pointer-events-none absolute inset-0 opacity-70" />
          <div className="relative flex h-[208px] flex-col">
            <div className="flex items-center justify-between px-4 pt-3.5">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    recorder.isRecording
                      ? "animate-blink bg-red-400"
                      : recorder.isBusy
                        ? "animate-blink bg-cyan-300"
                        : "bg-zinc-600",
                  )}
                />
                <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-500">
                  input · {status}
                </span>
              </div>
              <span className="max-w-[55%] truncate font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600">
                {recorder.capture?.deviceName || "default microphone"}
              </span>
            </div>
            <div className="min-h-0 flex-1 px-3 py-2">
              <Waveform
                levels={recorder.meter.levels}
                active={recorder.isRecording}
                processing={recorder.phase === "stopping"}
              />
            </div>
            <div className="flex items-end justify-between px-4 pb-3">
              <div className="font-mono text-[10px] text-zinc-600">
                {recorder.capture?.inputSampleRate
                  ? `${(recorder.capture.inputSampleRate / 1000).toFixed(1)} kHz in → 16 kHz`
                  : "VAD · AUTO TRIM"}
              </div>
              <div
                className={cn(
                  "font-mono text-[26px] font-medium leading-none tracking-tight tabular-nums",
                  recorder.isRecording ? "text-primary" : "text-zinc-500",
                )}
              >
                {formatTimer(recorder.meter.durationMs)}
                <span className="ml-2 align-middle text-[9px] uppercase tracking-[0.14em] text-zinc-600">
                  / {Math.round(AUTO_STOP_MS / 1000)}s
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between gap-4">
          <div className="w-40">
            <div className="label-caps">Live input</div>
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-1 w-14 overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-cyan-300/70 transition-[width] duration-100"
                  style={{ width: `${Math.min(100, recorder.meter.rms * 900)}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-zinc-600">
                {(recorder.meter.rms * 100).toFixed(1)}
              </span>
            </div>
          </div>

          <div className="flex flex-col items-center gap-2">
            <button
              type="button"
              onClick={() => void handleRecordClick()}
              disabled={pipelineBusy || recorder.isBusy}
              className={cn(
                "relative flex h-16 w-16 items-center justify-center rounded-full border transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-4 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-40",
                recorder.isRecording
                  ? "border-red-400/40 bg-red-400 text-red-950 shadow-[0_0_32px_-6px_rgba(248,113,113,0.55)] hover:bg-red-300"
                  : "border-primary/40 bg-primary text-primary-foreground shadow-[0_0_34px_-8px_hsl(78_82%_61%/0.6)] hover:bg-primary/90",
              )}
              aria-label={recorder.isRecording ? "Stop and trim recording" : "Start recording"}
            >
              {recorder.isRecording && (
                <span className="absolute inset-0 -z-10 animate-pulse-ring rounded-full border border-red-400/60" />
              )}
              {recorder.phase === "starting" || recorder.phase === "stopping" ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : recorder.isRecording ? (
                <Square className="h-5 w-5 fill-current" />
              ) : (
                <Mic className="h-6 w-6" />
              )}
            </button>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground">
              {recorder.isRecording ? "Stop & trim" : isWake ? "Record phrase" : "Record speech"}
            </span>
          </div>

          <div className="w-40 text-right">
            <div className="label-caps">Take guidance</div>
            <p className="mt-1.5 text-[10.5px] leading-relaxed text-zinc-600">
              {isWake ? "Natural pace, 1–3 words." : "Normal voice, up to 10 s."}
              <br />
              {isWake ? "20+ takes recommended." : "20–40 takes (~3 min)."}
            </p>
          </div>
        </div>

        {showError ? (
          <div className="mt-4 flex items-start gap-2.5 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2.5 text-xs text-red-200">
            <X className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="min-w-0 flex-1 leading-relaxed">{showError}</span>
            <button
              type="button"
              onClick={() => {
                recorder.clearError();
                setListError(null);
              }}
              className="shrink-0 text-red-300/50 hover:text-red-200"
              aria-label="Dismiss error"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : null}

        {recorder.lastResult ? (
          <div className="mt-4 flex items-center gap-2 rounded-md border border-primary/15 bg-primary/[0.035] px-3 py-2 text-[11px] text-muted-foreground">
            <Waves className="h-3.5 w-3.5 shrink-0 text-primary/80" />
            <span>
              Saved <span className="text-zinc-300">{formatDuration(recorder.lastResult.recording.durationMs)}</span>{" "}
              of speech
              {recorder.lastResult.trim.trimmedMs > 0
                ? ` · removed ${formatDuration(recorder.lastResult.trim.trimmedMs)} of silence`
                : ""}
              {recorder.lastResult.trim.resampledFrom !== 16000
                ? ` · resampled from ${(recorder.lastResult.trim.resampledFrom / 1000).toFixed(1)} kHz`
                : ""}
            </span>
          </div>
        ) : null}

        <div className="mt-5 border-t border-border/60 pt-4">
          <div className="mb-2.5 flex items-center justify-between">
            <div className="label-caps">{isWake ? "Clean takes" : "Other speech takes"}</div>
            <span className="font-mono text-[10px] text-zinc-600">
              {takes.length} files · WAV / PCM16
            </span>
          </div>

          {takes.length === 0 ? (
            <div className="flex items-center gap-4 rounded-lg border border-dashed border-border/70 bg-background/25 px-4 py-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-secondary/50">
                <AudioLines className="h-4 w-4 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xs font-medium text-zinc-300">
                  {isWake ? "No clean takes yet" : "No other speech yet"}
                </p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {isWake
                    ? "Record at least 3 takes to build a leakage-free dataset."
                    : "Optional, but the best defence against triggering on your own voice."}
                </p>
              </div>
            </div>
          ) : (
            <div className="max-h-[286px] space-y-1 overflow-y-auto pr-1">
              {takes.map((recording, index) => {
                const isPlaying = playingId === recording.id;
                const isLoading = loadingPlaybackId === recording.id;
                return (
                  <div
                    key={recording.id}
                    className={cn(
                      "group flex items-center gap-3 rounded-md border border-transparent px-2 py-2 transition-colors hover:border-border/70 hover:bg-secondary/30",
                      isPlaying && "border-primary/15 bg-primary/[0.035]",
                    )}
                  >
                    <span className="w-5 shrink-0 font-mono text-[9px] text-zinc-700">
                      {(index + 1).toString().padStart(2, "0")}
                    </span>
                    <button
                      type="button"
                      onClick={() => void handlePlay(recording)}
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-background text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={isPlaying ? "Pause take" : "Play take"}
                    >
                      {isLoading ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : isPlaying ? (
                        <Pause className="h-3 w-3 fill-current" />
                      ) : (
                        <Play className="h-3 w-3 fill-current pl-0.5" />
                      )}
                    </button>
                    <div className="hidden h-4 w-20 shrink-0 items-center gap-[2px] sm:flex">
                      {miniBars(recording.id).map((height, barIndex) => (
                        <span
                          key={barIndex}
                          className={cn(
                            "w-[2px] rounded-full",
                            isPlaying ? "bg-primary/70" : "bg-zinc-700",
                          )}
                          style={{ height: `${Math.max(2, height * 16)}px` }}
                        />
                      ))}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div
                        className={cn(
                          "truncate text-zinc-300",
                          recording.prompt ? "text-[12px]" : "font-mono text-[11px]",
                        )}
                        title={recording.prompt ?? recording.id}
                      >
                        {recording.prompt ?? recording.id}
                      </div>
                      <div className="mt-0.5 flex items-center gap-2 font-mono text-[9px] uppercase tracking-wider text-zinc-600">
                        <span>{formatDuration(recording.durationMs)}</span>
                        <span className="text-zinc-800">·</span>
                        <span>{formatDate(recording.createdAt)}</span>
                        <span className="hidden text-zinc-800 sm:inline">·</span>
                        <span className="hidden sm:inline">{formatBytes(recording.sizeBytes)}</span>
                      </div>
                    </div>
                    {recording.trimmedMs > 0 ? (
                      <span
                        className="hidden font-mono text-[9px] text-zinc-600 md:inline"
                        title="Silence removed by the VAD"
                      >
                        -{formatDuration(recording.trimmedMs)}
                      </span>
                    ) : null}
                    <div className="flex items-center opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                      {isWake ? (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-zinc-600 hover:text-zinc-200"
                          onClick={() => void handleRename(recording)}
                          disabled={recorder.phase !== "idle" || pipelineBusyRef.current || isPlaying}
                          aria-label="Rename take"
                        >
                          <Pencil className="h-3 w-3" />
                        </Button>
                      ) : null}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-zinc-600 hover:text-red-300"
                        onClick={() => void handleDelete(recording)}
                        disabled={recorder.phase !== "idle" || pipelineBusyRef.current || isPlaying}
                        aria-label="Delete take"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
