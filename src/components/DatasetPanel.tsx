import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  AudioLines,
  FolderOpen,
  Layers,
  Loader2,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import type { PipelineKind } from "@/hooks/useJobRunner";
import type { SpeechCorpus } from "@/lib/voice";
import { formatDate } from "@/lib/utils";
import type { DatasetSummary, JobProgress, ProjectSummary } from "@/types";

type DatasetPanelProps = {
  summary: ProjectSummary;
  corpora: SpeechCorpus[];
  onOpenSpeechData: () => void;
  environmentReady: boolean;
  jobBusy: boolean;
  jobKind: PipelineKind;
  progress: JobProgress | null;
  jobError: string | null;
  interactionDisabled: boolean;
  onStartAugment: (params: {
    project: string;
    variations: number;
    clipSeconds: number;
    backgroundDir: string | null;
    negativeDir: string | null;
    useSynthetic: boolean;
    useCorpora: boolean;
    corpusNegatives: number;
    useMined: boolean;
  }) => void;
  onDeleteMined: () => void;
  onCancelJob: () => void;
};

function countLabel(count: number) {
  return count.toString().padStart(2, "0");
}

const CORPUS_CLIP_OPTIONS = [2000, 6000, 12000, 20000];

// Takes are saved with 160 ms of pre/post roll that the augmenter trims again.
const TAKE_PADDING_MS = 320;
// Must match MAX_PHRASE_FILL in python/augment.py.
const MAX_PHRASE_FILL = 0.97;

type FolderFieldProps = {
  label: string;
  value: string | null;
  placeholder: string;
  title: string;
  disabled: boolean;
  onChoose: () => void;
  onClear: () => void;
};

function FolderField({ label, value, placeholder, title, disabled, onChoose, onClear }: FolderFieldProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        {value ? (
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            className="flex items-center gap-1 font-mono text-[9px] uppercase tracking-wider text-zinc-600 hover:text-zinc-300"
          >
            <X className="h-2.5 w-2.5" /> clear
          </button>
        ) : null}
      </div>
      <button
        type="button"
        onClick={onChoose}
        disabled={disabled}
        className="focus-ring flex h-9 w-full items-center gap-2 rounded-md border border-dashed border-border bg-background/30 px-3 text-left text-xs text-muted-foreground transition-colors hover:border-border hover:bg-secondary/30 disabled:cursor-not-allowed disabled:opacity-40"
        title={value ?? title}
      >
        <FolderOpen className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
        <span className="truncate">{value ?? placeholder}</span>
      </button>
    </div>
  );
}

export function DatasetPanel({
  summary,
  corpora,
  onOpenSpeechData,
  onDeleteMined,
  environmentReady,
  jobBusy,
  jobKind,
  progress,
  jobError,
  interactionDisabled,
  onStartAugment,
  onCancelJob,
}: DatasetPanelProps) {
  const [variations, setVariations] = useState(30);
  const [clipSeconds, setClipSeconds] = useState(1.5);
  const [backgroundDir, setBackgroundDir] = useState<string | null>(null);
  const [negativeDir, setNegativeDir] = useState<string | null>(null);
  const [useSynthetic, setUseSynthetic] = useState(true);
  const [useCorpora, setUseCorpora] = useState(true);
  const [corpusNegatives, setCorpusNegatives] = useState(6000);
  const [useMined, setUseMined] = useState(true);
  const minedActive = useMined && summary.minedCount > 0;
  const speechSets = corpora.filter((corpus) => corpus.role === "negatives" && corpus.installed);
  const speechHours = speechSets.reduce((sum, corpus) => sum + (corpus.installedHours ?? 0), 0);
  const corporaActive = useCorpora && speechSets.length > 0;
  const [localError, setLocalError] = useState<string | null>(null);
  const synthetic = summary.synthetic;
  const syntheticActive = Boolean(synthetic) && useSynthetic;
  const isAugmenting = jobBusy && jobKind === "augment";
  const dataset: DatasetSummary = summary.dataset;
  const canStart =
    environmentReady &&
    summary.recordings.length >= 3 &&
    !jobBusy &&
    !interactionDisabled &&
    variations >= 10;
  const totalVariations = summary.recordings.length * variations;
  const maxPhraseMs = clipSeconds * 1000 * MAX_PHRASE_FILL;
  const longTakes = summary.recordings.filter(
    (recording) => recording.durationMs - TAKE_PADDING_MS > maxPhraseMs,
  ).length;

  const chooseFolder = async (title: string, onSelected: (path: string) => void) => {
    setLocalError(null);
    try {
      const selected = await open({ directory: true, multiple: false, title });
      if (typeof selected === "string") onSelected(selected);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Could not open folder picker.");
    }
  };

  const start = () => {
    setLocalError(null);
    if (!Number.isFinite(variations) || variations < 10) {
      setLocalError("Use at least 10 variations per recording.");
      return;
    }
    if (dataset.exists) {
      const confirmed = window.confirm(
        "Rebuild the dataset?\n\nThe current augmented dataset will be replaced. Clean recordings and any trained model stay untouched.",
      );
      if (!confirmed) return;
    }
    onStartAugment({
      project: summary.slug,
      variations: Math.round(variations),
      clipSeconds,
      backgroundDir,
      negativeDir,
      useSynthetic: syntheticActive,
      useCorpora: corporaActive,
      corpusNegatives,
      useMined: minedActive,
    });
  };

  const error = localError || (jobKind === "augment" ? jobError : null);

  return (
    <section id="dataset" className="panel scroll-mt-24 p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">03</span>
            <span className="label-caps">Dataset</span>
          </div>
          <h2 className="mt-2 text-lg font-semibold tracking-tight">Expand the samples</h2>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Pitch, stretch, noise and gain variation with a source-level split.
          </p>
        </div>
        <Badge variant={dataset.exists ? "cyan" : "outline"}>
          {dataset.exists ? `${dataset.total} clips` : "not built"}
        </Badge>
      </div>

      <div className="mt-5 overflow-hidden rounded-lg border border-border/70 bg-background/30">
        <div className="grid grid-cols-[1fr_64px_64px] gap-2 border-b border-border/60 bg-white/[0.015] px-3 py-2 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-600">
          <span>Split</span>
          <span className="text-right">Wake</span>
          <span className="text-right">BG</span>
        </div>
        {(
          [
            ["train", dataset.train],
            ["val", dataset.val],
            ["test", dataset.test],
          ] as const
        ).map(([name, counts]) => (
          <div
            key={name}
            className="grid grid-cols-[1fr_64px_64px] gap-2 border-b border-border/30 px-3 py-2 last:border-0"
          >
            <span className="flex items-center gap-2 font-mono text-[10.5px] text-zinc-400">
              <span className="h-1 w-1 rounded-full bg-zinc-700" />
              {name}
            </span>
            <span className="text-right font-mono text-[11px] tabular-nums text-zinc-300">
              {countLabel(counts.wakeWord)}
            </span>
            <span className="text-right font-mono text-[11px] tabular-nums text-zinc-500">
              {countLabel(counts.background)}
            </span>
          </div>
        ))}
      </div>

      {dataset.exists ? (
        <div className="mt-2.5 flex items-center justify-between font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600">
          <span>
            {dataset.variationsPerSample ?? variations} variants / take · {dataset.clipSeconds ?? 1.5}s
          </span>
          <span className="flex items-center gap-1.5">
            {dataset.backgroundSource === "real" ? "real backgrounds" : "synthetic na"}
            <span className="h-1 w-1 rounded-full bg-zinc-700" />
            {dataset.createdAt ? formatDate(dataset.createdAt) : ""}
          </span>
        </div>
      ) : null}

      <div className="mt-5 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="variations">Variations / take</Label>
            <Input
              id="variations"
              type="number"
              min={10}
              max={40}
              value={variations}
              onChange={(event) => setVariations(Number(event.target.value))}
              disabled={jobBusy || interactionDisabled}
              className="font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="clip-length">Window</Label>
            <select
              id="clip-length"
              value={clipSeconds}
              onChange={(event) => setClipSeconds(Number(event.target.value))}
              disabled={jobBusy || interactionDisabled}
              className="focus-ring flex h-9 w-full rounded-md border border-input bg-background/60 px-3 py-1 font-mono text-xs shadow-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value={1.0}>1.0 s</option>
              <option value={1.5}>1.5 s</option>
              <option value={2.0}>2.0 s</option>
              <option value={2.5}>2.5 s</option>
            </select>
          </div>
        </div>

        <FolderField
          label="Backgrounds (optional)"
          value={backgroundDir}
          placeholder="Use synthetic room noise"
          title="Choose a folder of real background recordings"
          disabled={jobBusy || interactionDisabled}
          onChoose={() =>
            void chooseFolder("Choose a folder with background / noise WAV files", setBackgroundDir)
          }
          onClear={() => setBackgroundDir(null)}
        />

        <FolderField
          label="Negative phrases (optional)"
          value={negativeDir}
          placeholder="Other words, similar-sounding phrases"
          title="Recordings of other words and phrases (ideally similar to the wake phrase) that must not trigger"
          disabled={jobBusy || interactionDisabled}
          onChoose={() =>
            void chooseFolder("Choose a folder with recordings of other words / phrases", setNegativeDir)
          }
          onClear={() => setNegativeDir(null)}
        />

        {synthetic ? (
          <label className="flex items-center gap-2 rounded-md border border-border/50 bg-background/25 px-3 py-2 text-[11px] text-zinc-400">
            <input
              type="checkbox"
              checked={useSynthetic}
              onChange={(event) => setUseSynthetic(event.target.checked)}
              disabled={jobBusy || interactionDisabled}
              className="h-3.5 w-3.5 accent-[hsl(78_82%_61%)]"
            />
            <span>
              Use synthetic voices: <span className="text-zinc-200">{synthetic.counts.positive}</span> wake ·{" "}
              <span className="text-zinc-200">{synthetic.counts.negative}</span> negative clips
            </span>
          </label>
        ) : null}

        {speechSets.length ? (
          <div className="flex flex-wrap items-center gap-2 rounded-md border border-border/50 bg-background/25 px-3 py-2 text-[11px] text-zinc-400">
            <label className="flex min-w-0 flex-1 items-center gap-2">
              <input
                type="checkbox"
                checked={useCorpora}
                onChange={(event) => setUseCorpora(event.target.checked)}
                disabled={jobBusy || interactionDisabled}
                className="h-3.5 w-3.5 accent-[hsl(78_82%_61%)]"
              />
              <span className="min-w-0">
                Real speech: <span className="text-zinc-200">{speechHours.toFixed(1)} h</span> from{" "}
                {speechSets.map((corpus) => corpus.title).join(", ")}
              </span>
            </label>
            <select
              value={corpusNegatives}
              onChange={(event) => setCorpusNegatives(Number(event.target.value))}
              disabled={jobBusy || interactionDisabled || !useCorpora}
              className="focus-ring h-7 rounded-md border border-input bg-background/60 px-2 font-mono text-[10.5px] disabled:opacity-50"
              aria-label="Speech clips"
            >
              {CORPUS_CLIP_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value.toLocaleString()} clips
                </option>
              ))}
            </select>
          </div>
        ) : (
          <button
            type="button"
            onClick={onOpenSpeechData}
            className="flex w-full items-start gap-2 rounded-md border border-amber-400/20 bg-amber-400/[0.045] px-3 py-2 text-left text-[10.5px] leading-relaxed text-amber-100/70 hover:bg-amber-400/[0.07]"
          >
            <AudioLines className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300/80" />
            <span>
              No real speech negatives yet: without them the model tends to fire on any voice.
              Download a speech set in AI models → Speech data.
            </span>
          </button>
        )}

        {summary.minedCount > 0 ? (
          <div className="flex items-center gap-2 rounded-md border border-border/50 bg-background/25 px-3 py-2 text-[11px] text-zinc-400">
            <label className="flex min-w-0 flex-1 items-center gap-2">
              <input
                type="checkbox"
                checked={useMined}
                onChange={(event) => setUseMined(event.target.checked)}
                disabled={jobBusy || interactionDisabled}
                className="h-3.5 w-3.5 accent-[hsl(78_82%_61%)]"
              />
              <span>
                Mined hard negatives: <span className="text-zinc-200">{summary.minedCount}</span> windows an
                earlier model fired on (train only)
              </span>
            </label>
            <button
              type="button"
              onClick={() => {
                if (window.confirm("Forget all mined hard negatives of this project?")) onDeleteMined();
              }}
              disabled={jobBusy || interactionDisabled}
              className="font-mono text-[9px] uppercase tracking-wider text-zinc-600 hover:text-red-300 disabled:opacity-40"
            >
              forget
            </button>
          </div>
        ) : null}

        {longTakes > 0 ? (
          <div className="flex gap-2 rounded-md border border-amber-400/20 bg-amber-400/[0.045] px-3 py-2 text-[10.5px] leading-relaxed text-amber-100/70">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300/80" />
            <span>
              {longTakes} take{longTakes === 1 ? "" : "s"} may not fit the {clipSeconds.toFixed(1)} s
              window and will be skipped. Choose a longer window or re-record them.
            </span>
          </div>
        ) : null}

        {dataset.exists && dataset.warnings.length > 0 ? (
          <div className="space-y-1 rounded-md border border-amber-400/20 bg-amber-400/[0.045] px-3 py-2 text-[10.5px] leading-relaxed text-amber-100/70">
            {dataset.warnings.map((warning) => (
              <p key={warning} className="flex gap-2">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300/80" />
                <span>{warning}</span>
              </p>
            ))}
          </div>
        ) : null}

        <div className="rounded-md border border-border/50 bg-background/25 px-3 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
          {summary.recordings.length < 3 ? (
            <span className="flex gap-2">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300/80" />
              Record at least 3 takes. The source-level split prevents augmented twins from leaking
              into validation.
            </span>
          ) : (
            <span className="flex gap-2">
              <Layers className="mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan-300/60" />
              {summary.recordings.length} source takes × {variations} variants ≈{" "}
              <span className="text-zinc-300">{totalVariations} clips</span> plus as many
              negatives: noise, partial and reversed phrases
              {negativeDir ? ", other phrases" : ""}
              {summary.negativeTakes.length ? `, ${summary.negativeTakes.length} of your other-speech takes` : ""}
              {corporaActive ? `, plus ${corpusNegatives.toLocaleString()} real speech clips` : ""}
              {minedActive ? ` and ${summary.minedCount} mined hard negatives` : ""}
              {syntheticActive ? ", synthetic voices" : ""}
              {backgroundDir ? ", real backgrounds" : ""}.
            </span>
          )}
        </div>

        {error ? (
          <div className="rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[11px] leading-relaxed text-red-200">
            {error}
          </div>
        ) : null}

        {isAugmenting ? (
          <div className="space-y-2 rounded-md border border-cyan-300/15 bg-cyan-300/[0.035] px-3 py-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-2 text-[11px] text-cyan-100/80">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                <span className="truncate">{progress?.message || "Preparing augmenter…"}</span>
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 shrink-0 px-2 text-[10px] text-muted-foreground hover:text-red-300"
                onClick={onCancelJob}
              >
                <Square className="h-2.5 w-2.5" />
                Stop
              </Button>
            </div>
            <Progress
              value={(progress?.progress ?? 0) * 100}
              indicatorClassName="bg-cyan-300"
              className="h-1"
            />
          </div>
        ) : (
          <Button
            className="w-full"
            onClick={start}
            disabled={!canStart}
            title={
              !environmentReady
                ? "Set up the Python environment first"
                : summary.recordings.length < 3
                  ? "Record at least 3 takes"
                  : jobBusy
                    ? "A pipeline job is running"
                    : undefined
            }
          >
            {jobBusy ? (
              <Loader2 className="animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {dataset.exists ? "Rebuild dataset" : "Build dataset"}
          </Button>
        )}

        {!environmentReady ? (
          <p className="text-center font-mono text-[9px] uppercase tracking-[0.12em] text-amber-300/60">
            Python ML environment required
          </p>
        ) : null}
      </div>
    </section>
  );
}
