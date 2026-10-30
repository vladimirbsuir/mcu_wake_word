import { useMemo, useState } from "react";
import {
  AlertTriangle,
  AudioLines,
  Loader2,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Trash2,
  Wand2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import type { PipelineKind } from "@/hooks/useJobRunner";
import { useProjectVoice } from "@/hooks/useProjectVoice";
import { errorMessage } from "@/lib/api";
import { cn, formatDate } from "@/lib/utils";
import {
  categoryReadiness,
  choiceReadiness,
  isUsable,
  LOCAL_PROVIDER,
  playWav,
  sameChoice,
  ttsCandidates,
  ttsVoiceCount,
  VOICE_CATEGORIES,
  voiceApi,
  type SyntheticClip,
  type VoiceChoice,
  type VoiceCategory,
  type VoiceSettings,
  type VoiceStatus,
} from "@/lib/voice";
import type { JobFinishedEvent, JobProgress, JobStarted, ProjectSummary } from "@/types";

const LAB_JOBS: PipelineKind[] = ["words", "synthesize"];

type PhraseLabPanelProps = {
  summary: ProjectSummary;
  status: VoiceStatus | null;
  settings: VoiceSettings | null;
  jobBusy: boolean;
  jobKind: PipelineKind;
  jobProject: string | null;
  jobProgress: JobProgress | null;
  jobFinished: JobFinishedEvent | null;
  jobError: string | null;
  interactionDisabled: boolean;
  onRunJob: (kind: Exclude<PipelineKind, null>, starter: () => Promise<JobStarted>) => void;
  onCancelJob: () => void;
  onOpenModels: (category: VoiceCategory) => void;
  onRefresh: () => void | Promise<void>;
};

function GroupEditor({
  title,
  description,
  items,
  count,
  disabled,
  onCount,
  onChange,
}: {
  title: string;
  description: string;
  items: string[];
  count: number;
  disabled: boolean;
  onCount: (value: number) => void;
  onChange: (items: string[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const values = draft
      .split(/[,\n;]/)
      .map((value) => value.trim())
      .filter(Boolean)
      .filter((value) => !items.some((item) => item.toLowerCase() === value.toLowerCase()));
    if (values.length) onChange([...items, ...values]);
    setDraft("");
  };
  return (
    <div className="rounded-lg border border-border/60 bg-background/25 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[12.5px] font-medium text-zinc-200">
            {title}
            <span className="font-mono text-[9.5px] text-zinc-600">{items.length}</span>
          </div>
          <p className="mt-0.5 text-[10.5px] leading-relaxed text-muted-foreground">{description}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <Label className="text-[9px]">+ ask</Label>
          <Input
            type="number"
            min={0}
            max={200}
            value={count}
            onChange={(event) => onCount(Math.max(0, Math.min(200, Number(event.target.value) || 0)))}
            disabled={disabled}
            className="h-7 w-16 font-mono text-xs"
          />
          {items.length ? (
            <button
              type="button"
              onClick={() => onChange([])}
              disabled={disabled}
              className="rounded p-1 text-zinc-600 hover:text-red-300 disabled:opacity-40"
              title="Clear this list"
              aria-label={`Clear ${title}`}
            >
              <Trash2 className="h-3 w-3" />
            </button>
          ) : null}
        </div>
      </div>
      <div className="mt-2 flex max-h-40 flex-wrap gap-1 overflow-y-auto">
        {items.map((item) => (
          <span
            key={item}
            className="group inline-flex items-center gap-1 rounded border border-border/60 bg-secondary/30 py-0.5 pl-2 pr-1 text-[11px] text-zinc-300"
          >
            {item}
            <button
              type="button"
              onClick={() => onChange(items.filter((value) => value !== item))}
              disabled={disabled}
              className="rounded text-zinc-600 hover:text-red-300"
              aria-label={`Remove ${item}`}
            >
              <X className="h-2.5 w-2.5" />
            </button>
          </span>
        ))}
        {!items.length ? <span className="text-[10.5px] text-zinc-600">Empty. Generate or type your own.</span> : null}
      </div>
      <div className="mt-2 flex gap-1.5">
        <Input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") add();
          }}
          placeholder="Add words, comma-separated"
          disabled={disabled}
          className="h-7 text-xs"
        />
        <Button variant="outline" size="sm" className="h-7 px-2" onClick={add} disabled={disabled || !draft.trim()}>
          <Plus className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}

export function PhraseLabPanel({
  summary,
  status,
  settings,
  jobBusy,
  jobKind,
  jobProject,
  jobProgress,
  jobFinished,
  jobError,
  interactionDisabled,
  onRunJob,
  onCancelJob,
  onOpenModels,
  onRefresh,
}: PhraseLabPanelProps) {
  const project = summary.slug;
  const { voice, spokenText, language, update, flush, saveError } = useProjectVoice();
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [playing, setPlaying] = useState<string | null>(null);

  const groups = status?.groups ?? [];
  const languageInfo = status?.languages.find((item) => item.code === language) ?? null;
  const readiness = useMemo(
    () =>
      Object.fromEntries(
        VOICE_CATEGORIES.map((item) => [item.id, categoryReadiness(item.id, language, settings, status)]),
      ) as Record<VoiceCategory, ReturnType<typeof categoryReadiness>>,
    [language, settings, status],
  );
  const labJobRunning = jobBusy && LAB_JOBS.includes(jobKind) && jobProject === project;
  const disabled = jobBusy || interactionDisabled;

  const start = async (kind: Exclude<PipelineKind, null>, starter: () => Promise<JobStarted>) => {
    // Jobs read voice.json from disk.
    await flush();
    onRunJob(kind, starter);
  };

  const groupCount = (id: string, fallback: number) => counts[id] ?? fallback;
  const requested = Object.fromEntries(groups.map((group) => [group.id, groupCount(group.id, group.defaultCount)]));
  const negativeItems = Object.values(voice.groups ?? {}).reduce((sum, group) => sum + (group.items?.length ?? 0), 0);
  const synthesis = voice.synthesis ?? { voicesPerNegative: 2, includePositives: true };
  const activeTts = settings?.active?.tts ?? null;
  const engines: VoiceChoice[] = synthesis.engines ?? (activeTts?.provider ? [activeTts] : []);
  const candidates = ttsCandidates(settings, status);
  const engineReadiness = (choice: VoiceChoice) => choiceReadiness("tts", choice, language, settings, status);
  const engineName = (choice: VoiceChoice) =>
    choice.provider === LOCAL_PROVIDER
      ? (status?.models.find((item) => item.id === choice.model)?.title ?? choice.model ?? "")
      : (settings?.profiles.find((item) => item.id === choice.provider)?.name ?? choice.provider);
  const enginesReady = engines.length > 0 && engines.every((choice) => isUsable(engineReadiness(choice)));
  // "Voices per word" is a total over all ticked engines; every voice says the phrase.
  const engineVoices = engines.map((choice) => ttsVoiceCount(choice, settings, status));
  const estimate =
    negativeItems * Math.min(synthesis.voicesPerNegative, engineVoices.reduce((sum, value) => sum + value, 0)) +
    (synthesis.includePositives ? engineVoices.reduce((sum, value) => sum + Math.min(value, 40) * 3, 0) : 0);
  const toggleEngine = (choice: VoiceChoice) =>
    update((current) => {
      const currentSynthesis = current.synthesis ?? synthesis;
      const selected = currentSynthesis.engines ?? engines;
      const next = selected.some((item) => sameChoice(item, choice))
        ? selected.filter((item) => !sameChoice(item, choice))
        : [...selected, choice];
      return { ...current, synthesis: { ...currentSynthesis, engines: next } };
    });
  const synthetic = summary.synthetic;

  const play = async (clip: SyntheticClip) => {
    setPlaying(clip.file);
    try {
      await playWav(await voiceApi.readSyntheticAudio(project, clip.file));
    } catch {
      // A missing clip just does not play.
    } finally {
      setPlaying(null);
    }
  };

  const labError = jobFinished && !jobFinished.success && jobFinished.project === project && LAB_JOBS.includes(jobFinished.kind as PipelineKind)
    ? (jobError ?? jobFinished.message)
    : null;

  return (
    <section id="phrases" className="panel scroll-mt-24 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">02</span>
            <span className="label-caps">Phrases</span>
          </div>
          <h2 className="mt-2 text-lg font-semibold tracking-tight">Teach it what not to hear</h2>
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted-foreground">
            A language model writes similar-sounding words and everyday phrases; a TTS voices them,
            and your wake phrase, in many synthetic voices. Optional, everything stays editable.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => onOpenModels("llm")}>
          <Settings2 className="h-3.5 w-3.5" /> AI models
        </Button>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-3">
          <p className="text-[10.5px] leading-relaxed text-zinc-600">
            Lists and voices are made for “<span className="text-zinc-300">{spokenText}</span>” in{" "}
            {languageInfo?.name ?? language}. Change the phrase in the bar at the top.
          </p>

          <div className="grid grid-cols-3 gap-2">
            {VOICE_CATEGORIES.map((item) => {
              const state = readiness[item.id];
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onOpenModels(item.id)}
                  className="rounded-md border border-border/60 bg-background/25 px-2.5 py-2 text-left transition-colors hover:border-border"
                  title={state.provider ?? "Choose a provider"}
                >
                  <div className="flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-wider text-zinc-500">
                    <span
                      className={cn(
                        "h-1.5 w-1.5 rounded-full",
                        isUsable(state) ? "bg-primary" : state.state === "unselected" ? "bg-zinc-600" : "bg-amber-400",
                      )}
                    />
                    {item.short}
                  </div>
                  <div className="mt-1 truncate text-[11px] text-zinc-300">{state.provider ?? "not selected"}</div>
                  <div
                    className={cn(
                      "truncate font-mono text-[9px]",
                      state.state === "unsupported" ? "text-red-300/80" : "text-zinc-600",
                    )}
                  >
                    {state.state === "unsupported" ? `no ${languageInfo?.name ?? language}` : state.label}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="rounded-lg border border-border/60 bg-background/25 p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[12.5px] font-medium text-zinc-200">
                <AudioLines className="h-3.5 w-3.5 text-zinc-500" /> Synthetic voices
              </div>
              <Button
                size="sm"
                className="h-7"
                disabled={disabled || !enginesReady || (!negativeItems && !synthesis.includePositives)}
                title={!enginesReady ? "Choose at least one ready TTS" : "Only clips that are not cached yet are synthesized"}
                onClick={() =>
                  void start("synthesize", () =>
                    voiceApi.synthesize(project, synthesis.includePositives, synthesis.voicesPerNegative),
                  )
                }
              >
                {labJobRunning && jobKind === "synthesize" ? <Loader2 className="animate-spin" /> : <Sparkles className="h-3 w-3" />}
                Synthesize{estimate ? ` ~${estimate} clips` : ""}
              </Button>
            </div>
            <div className="mt-2 space-y-1">
              <Label>Voices from</Label>
              <div className="flex flex-wrap gap-1.5">
                {candidates.map((choice) => {
                  const state = engineReadiness(choice);
                  const usable = isUsable(state);
                  const checked = engines.some((item) => sameChoice(item, choice));
                  return (
                    <label
                      key={`${choice.provider}:${choice.model ?? ""}`}
                      className={cn(
                        "flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px]",
                        checked ? "border-primary/40 bg-primary/[0.05] text-zinc-200" : "border-border/60 text-zinc-400",
                        !usable && !checked && "opacity-50",
                      )}
                      title={usable ? `${ttsVoiceCount(choice, settings, status)} voices` : state.label}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleEngine(choice)}
                        disabled={disabled || (!usable && !checked)}
                        className="h-3 w-3 accent-[hsl(78_82%_61%)]"
                      />
                      {engineName(choice)}
                      <span className="font-mono text-[9px] text-zinc-600">
                        {usable ? ttsVoiceCount(choice, settings, status) : state.label}
                      </span>
                    </label>
                  );
                })}
                <button
                  type="button"
                  onClick={() => onOpenModels("tts")}
                  className="rounded-md px-1.5 text-[10.5px] text-zinc-600 hover:text-zinc-300"
                >
                  more…
                </button>
              </div>
              {engines.some((choice) => !isUsable(engineReadiness(choice))) ? (
                <p className="text-[10px] text-amber-200/70">
                  A selected TTS is not ready ({engines
                    .filter((choice) => !isUsable(engineReadiness(choice)))
                    .map(engineName)
                    .join(", ")}): download it or untick it.
                </p>
              ) : null}
              <p className="text-[10px] text-zinc-600">
                Clips are cached per TTS and voice: only new words, voices or engines are synthesized,
                and the dataset uses clips of the ticked engines only.
              </p>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2.5">
              <div className="space-y-1">
                <Label htmlFor="voices-per-word">Voices per word (all engines)</Label>
                <select
                  id="voices-per-word"
                  value={synthesis.voicesPerNegative}
                  onChange={(event) =>
                    update((current) => ({
                      ...current,
                      synthesis: { ...synthesis, voicesPerNegative: Number(event.target.value) },
                    }))
                  }
                  disabled={disabled}
                  className="focus-ring flex h-8 w-full rounded-md border border-input bg-background/60 px-2 font-mono text-xs"
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <label className="flex items-end gap-2 pb-1.5 text-[11px] text-zinc-400">
                <input
                  type="checkbox"
                  checked={synthesis.includePositives}
                  onChange={(event) =>
                    update((current) => ({
                      ...current,
                      synthesis: { ...synthesis, includePositives: event.target.checked },
                    }))
                  }
                  disabled={disabled}
                  className="h-3.5 w-3.5 accent-[hsl(78_82%_61%)]"
                />
                Also voice the wake phrase (train only)
              </label>
            </div>
            {synthetic ? (
              <div className="mt-3 space-y-2 border-t border-border/50 pt-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2 text-[10.5px] text-zinc-500">
                  <span>
                    <span className="text-zinc-300">{synthetic.counts.positive}</span> wake ·{" "}
                    <span className="text-zinc-300">{synthetic.counts.negative}</span> negative · {synthetic.voices} voices ·{" "}
                    {synthetic.engine} · {formatDate(synthetic.createdAt)}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm("Delete all synthetic clips of this project?")) {
                        void voiceApi
                          .deleteSynthetic(project)
                          .then(() => onRefresh())
                          .catch((error) => window.alert(errorMessage(error)));
                      }
                    }}
                    disabled={disabled}
                    className="flex items-center gap-1 text-zinc-600 hover:text-red-300"
                  >
                    <Trash2 className="h-3 w-3" /> delete
                  </button>
                </div>
                {(["positive", "negative"] as const).map((label) =>
                  synthetic.samples[label]?.length ? (
                    <div key={label} className="flex flex-wrap gap-1">
                      {synthetic.samples[label]!.slice(0, label === "positive" ? 12 : 8).map((clip) => (
                        <button
                          key={clip.file}
                          type="button"
                          onClick={() => void play(clip)}
                          disabled={playing !== null}
                          title={`${clip.text} · ${clip.voiceName} · ×${clip.speed}`}
                          className={cn(
                            "inline-flex max-w-[160px] items-center gap-1 rounded border px-1.5 py-0.5 text-[10px]",
                            label === "positive"
                              ? "border-primary/25 text-primary/80"
                              : "border-border/60 text-zinc-400",
                            playing === clip.file && "bg-white/[0.06]",
                          )}
                        >
                          {playing === clip.file ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : <Play className="h-2.5 w-2.5" />}
                          <span className="truncate">{label === "positive" ? clip.voiceName : clip.text}</span>
                        </button>
                      ))}
                    </div>
                  ) : null,
                )}
                <p className="text-[10px] text-zinc-600">
                  Used by the next dataset build: wake-phrase voices go to train only, negatives are
                  split by text across train/val/test.
                </p>
              </div>
            ) : null}
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <span className="label-caps">Negative word lists</span>
            <Button
              size="sm"
              variant="outline"
              className="h-7"
              disabled={
                disabled ||
                !isUsable(readiness.llm) ||
                !spokenText ||
                Object.values(requested).every((value) => value === 0)
              }
              title={!isUsable(readiness.llm) ? "Set up a language model first" : undefined}
              onClick={() => void start("words", () => voiceApi.generateWords(project, requested))}
            >
              {labJobRunning && jobKind === "words" ? <Loader2 className="animate-spin" /> : <Wand2 className="h-3 w-3" />}
              Generate
            </Button>
          </div>
          {groups.map((group) => (
            <GroupEditor
              key={group.id}
              title={group.title}
              description={group.description}
              items={voice.groups?.[group.id]?.items ?? []}
              count={groupCount(group.id, group.defaultCount)}
              disabled={disabled}
              onCount={(value) => setCounts((current) => ({ ...current, [group.id]: value }))}
              onChange={(items) =>
                update((current) => ({
                  ...current,
                  groups: { ...(current.groups ?? {}), [group.id]: { ...(current.groups?.[group.id] ?? {}), items } },
                }))
              }
            />
          ))}
          {!status ? (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
            </div>
          ) : null}
        </div>
      </div>

      {labJobRunning ? (
        <div className="mt-4 space-y-2 rounded-md border border-cyan-300/15 bg-cyan-300/[0.035] px-3 py-2.5">
          <div className="flex items-center justify-between gap-2 text-[11px] text-cyan-100/80">
            <span className="flex min-w-0 items-center gap-2">
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
              <span className="truncate">{jobProgress?.message ?? "Starting…"}</span>
            </span>
            <Button variant="ghost" size="sm" className="h-6 px-2 text-[10px]" onClick={onCancelJob}>
              Stop
            </Button>
          </div>
          <Progress value={(jobProgress?.progress ?? 0) * 100} className="h-1" indicatorClassName="bg-cyan-300" />
        </div>
      ) : null}

      {labError || saveError ? (
        <div className="mt-4 flex gap-2 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[11px] leading-relaxed text-red-200">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{labError ?? saveError}</span>
        </div>
      ) : null}
    </section>
  );
}
