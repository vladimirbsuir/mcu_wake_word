import { useState } from "react";
import {
  Activity,
  BrainCircuit,
  Cpu,
  Gauge,
  Loader2,
  Play,
  Square,
  Target,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { LogConsole } from "@/components/LogConsole";
import { MetricChart } from "@/components/MetricChart";
import { StreamingTest } from "@/components/StreamingTest";
import type { LogLine, PipelineKind } from "@/hooks/useJobRunner";
import { cn, formatBytes, percent } from "@/lib/utils";
import type { SpeechCorpus } from "@/lib/voice";
import type {
  EpochMetrics,
  ModelArchitecture,
  JobFinishedEvent,
  JobProgress,
  ProjectSummary,
} from "@/types";

type TrainingPanelProps = {
  summary: ProjectSummary;
  corpora: SpeechCorpus[];
  onOpenSpeechData: () => void;
  onStartEvaluation: () => void;
  onStartMining: () => void;
  environmentReady: boolean;
  jobBusy: boolean;
  jobKind: PipelineKind;
  progress: JobProgress | null;
  liveHistory: EpochMetrics[];
  logs: LogLine[];
  finished: JobFinishedEvent | null;
  jobError: string | null;
  interactionDisabled: boolean;
  onStartTraining: (params: {
    project: string;
    epochs: number;
    batchSize: number;
    learningRate: number;
    architecture: ModelArchitecture;
  }) => void;
  onCancelJob: () => void;
  onClearLogs: () => void;
};

const DEFAULT_ARCHITECTURE: ModelArchitecture = "ds-cnn-temporal";

function latestMetric(epochs: EpochMetrics[], key: "val_accuracy" | "accuracy"): number | null {
  for (let index = epochs.length - 1; index >= 0; index -= 1) {
    const value = epochs[index][key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toFixed(digits);
}

function MetricTile({
  label,
  value,
  hint,
  icon: Icon,
  accent,
}: {
  label: string;
  value: string;
  hint: string;
  icon: typeof Target;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border/70 bg-background/35 px-3 py-2.5">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-600">
          {label}
        </span>
        <Icon className={cn("h-3 w-3", accent ? "text-primary/70" : "text-zinc-700")} />
      </div>
      <div
        className={cn(
          "mt-2 font-mono text-xl leading-none tracking-tight",
          accent ? "text-primary" : "text-zinc-200",
        )}
      >
        {value}
      </div>
      <div className="mt-1 text-[9.5px] text-zinc-600">{hint}</div>
    </div>
  );
}

export function TrainingPanel({
  summary,
  corpora,
  onOpenSpeechData,
  onStartEvaluation,
  onStartMining,
  environmentReady,
  jobBusy,
  jobKind,
  progress,
  liveHistory,
  logs,
  finished,
  jobError,
  interactionDisabled,
  onStartTraining,
  onCancelJob,
  onClearLogs,
}: TrainingPanelProps) {
  const [epochs, setEpochs] = useState(40);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [architecture, setArchitecture] = useState<ModelArchitecture>(DEFAULT_ARCHITECTURE);

  const isTraining = jobBusy && jobKind === "train";
  const history = isTraining ? liveHistory : (summary.model.history?.epochs ?? []);
  const metadata = summary.model.metadata;
  const modelMetrics = metadata?.metrics;
  const lastVal = latestMetric(liveHistory, "val_accuracy");
  const canStart =
    environmentReady &&
    summary.dataset.exists &&
    !jobBusy &&
    !interactionDisabled &&
    summary.dataset.train.wakeWord + summary.dataset.train.background > 0;

  const chartSeries = [
    {
      label: "train",
      color: "#c8f560",
      values: history.map((epoch) => epoch.accuracy),
    },
    {
      label: "val",
      color: "#67e8f9",
      values: history.map((epoch) => epoch.val_accuracy),
    },
  ];
  const lossSeries = [
    {
      label: "train",
      color: "#fbbf24",
      values: history.map((epoch) => epoch.loss),
    },
    {
      label: "val",
      color: "#f472b6",
      values: history.map((epoch) => epoch.val_loss),
    },
  ];

  const start = () => {
    onStartTraining({
      project: summary.slug,
      epochs: Math.round(epochs),
      batchSize,
      learningRate,
      architecture,
    });
  };

  const percentDone = Math.round((progress?.progress ?? 0) * 100);
  const lastRun = metadata?.training;
  const trainError = jobKind === "train" ? jobError : null;

  return (
    <section id="train" className="panel scroll-mt-24 overflow-hidden">
      <div className="border-b border-border/60 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">04</span>
              <span className="label-caps">Train</span>
            </div>
            <h2 className="mt-2 text-lg font-semibold tracking-tight">Shape the embedded model</h2>
            <p className="mt-1 max-w-xl text-xs leading-relaxed text-muted-foreground">
              A compact depthwise-separable CNN trained on log-Mel features, then fully quantized to
              INT8.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">DS-CNN</Badge>
            <Badge variant="outline">INT8</Badge>
          </div>
        </div>
      </div>

      <div className="grid gap-5 p-5 lg:grid-cols-[285px_minmax(0,1fr)]">
        <div className="flex flex-col rounded-lg border border-border/70 bg-background/30 p-4">
          <div className="label-caps">Training recipe</div>

          <div className="mt-4 space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="epochs">Epochs</Label>
                <span className="font-mono text-[11px] text-zinc-300">{epochs}</span>
              </div>
              <input
                id="epochs"
                type="range"
                min={10}
                max={120}
                step={5}
                value={Math.min(120, Math.max(10, epochs))}
                onChange={(event) => setEpochs(Number(event.target.value))}
                disabled={jobBusy || interactionDisabled}
                className="h-1 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-[#c8f560] disabled:cursor-not-allowed disabled:opacity-40"
              />
              <div className="flex justify-between font-mono text-[9px] text-zinc-700">
                <span>10</span>
                <span>Early stopping enabled</span>
                <span>120</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="batch-size">Batch</Label>
                <select
                  id="batch-size"
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                  disabled={jobBusy || interactionDisabled}
                  className="focus-ring flex h-9 w-full rounded-md border border-input bg-background/60 px-2.5 font-mono text-xs disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="learning-rate">LR</Label>
                <select
                  id="learning-rate"
                  value={learningRate}
                  onChange={(event) => setLearningRate(Number(event.target.value))}
                  disabled={jobBusy || interactionDisabled}
                  className="focus-ring flex h-9 w-full rounded-md border border-input bg-background/60 px-2.5 font-mono text-xs disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value={0.0005}>0.0005</option>
                  <option value={0.001}>0.001</option>
                  <option value={0.002}>0.002</option>
                </select>
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="architecture">Classifier head</Label>
              <select
                id="architecture"
                value={architecture}
                onChange={(event) => setArchitecture(event.target.value as ModelArchitecture)}
                disabled={jobBusy || interactionDisabled}
                className="focus-ring flex h-9 w-full rounded-md border border-input bg-background/60 px-2.5 font-mono text-xs disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="ds-cnn-temporal">Temporal · ~100k params</option>
                <option value="ds-cnn">Average pool · ~27k params</option>
              </select>
            </div>

            <div className="flex items-center justify-between rounded-md border border-border/60 bg-background/40 px-3 py-2 font-mono text-[9.5px] text-zinc-500">
              <span>val / test split</span>
              <span className="text-zinc-300">
                {summary.dataset.val.wakeWord + summary.dataset.val.background} /{" "}
                {summary.dataset.test.wakeWord + summary.dataset.test.background}
              </span>
            </div>
          </div>

          <div className="mt-auto pt-5">
            {isTraining ? (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between font-mono text-[10px] text-zinc-500">
                  <span className="flex items-center gap-2 text-cyan-100/80">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {progress?.epoch && progress?.epochs
                      ? `EPOCH ${String(progress.epoch).padStart(2, "0")} / ${progress.epochs}`
                      : "STARTING"}
                  </span>
                  <span>{percentDone}%</span>
                </div>
                <Progress value={percentDone} indicatorClassName="bg-cyan-300" className="h-1" />
                <div className="flex items-center justify-between gap-2">
                  <span className="min-w-0 truncate text-[10.5px] text-muted-foreground">
                    {progress?.message || "Launching TensorFlow…"}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 shrink-0 px-2 text-[10px] text-muted-foreground hover:text-red-300"
                    onClick={onCancelJob}
                  >
                    <Square className="h-2.5 w-2.5" />
                    Stop
                  </Button>
                </div>
              </div>
            ) : (
              <Button
                className="w-full"
                onClick={start}
                disabled={!canStart}
                title={
                  !environmentReady
                    ? "Set up the Python environment first"
                    : !summary.dataset.exists
                      ? "Build the dataset first"
                      : jobBusy
                        ? "A pipeline job is running"
                        : undefined
                }
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                {summary.model.exists ? "Retrain model" : "Train model"}
              </Button>
            )}
            <p className="mt-2.5 text-center font-mono text-[8.5px] uppercase leading-relaxed tracking-[0.1em] text-zinc-700">
              Local Python process · early stopping
              <br />
              Best checkpoint restored
            </p>
          </div>
        </div>

        <div className="min-w-0">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricTile
              label="Test acc"
              value={percent(modelMetrics?.test_accuracy ?? null)}
              hint="float model"
              icon={Target}
              accent
            />
            <MetricTile
              label="INT8 acc"
              value={percent(modelMetrics?.tflite_test_accuracy ?? null)}
              hint="quantized model"
              icon={Cpu}
              accent
            />
            <MetricTile
              label="F1"
              value={modelMetrics?.test_f1?.toFixed(3) ?? "—"}
              hint="wake-word class"
              icon={Activity}
            />
            <MetricTile
              label={isTraining ? "Live val" : "Best val"}
              value={isTraining ? percent(lastVal) : percent(lastRun?.best_val_accuracy ?? null)}
              hint={isTraining ? "current epoch" : lastRun ? `epoch ${lastRun.best_epoch}` : "not trained"}
              icon={Gauge}
            />
          </div>

          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <MetricChart
              title="Accuracy"
              series={chartSeries}
              yMin={0}
              yMax={1}
              formatValue={(value) => value.toFixed(2)}
            />
            <MetricChart
              title="Loss"
              series={lossSeries}
              yMin={0}
              formatValue={(value) => (value < 0.1 ? value.toFixed(3) : value.toFixed(2))}
            />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-md border border-border/50 bg-background/25 px-3 py-2 font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600">
            <span className="flex items-center gap-1.5">
              <BrainCircuit className="h-3 w-3 text-zinc-700" />
              {metadata ? `${formatNumber(metadata.model.parameters)} params` : "architecture pending"}
            </span>
            <span className="h-1 w-1 rounded-full bg-zinc-800" />
            <span>{summary.model.tfliteSizeBytes ? formatBytes(summary.model.tfliteSizeBytes) : "—"}</span>
            <span className="h-1 w-1 rounded-full bg-zinc-800" />
            <span>
              {metadata
                ? `[${metadata.input.shape.join("×")}] ${metadata.input.dtype.toUpperCase()}`
                : "input pending"}
            </span>
            {lastRun ? (
              <>
                <span className="h-1 w-1 rounded-full bg-zinc-800" />
                <span>{lastRun.epochs_completed} epochs</span>
              </>
            ) : null}
          </div>

          {trainError ? (
            <div className="mt-3 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[11px] leading-relaxed text-red-200">
              {trainError}
            </div>
          ) : null}

          {!isTraining && finished?.kind === "train" && finished.success ? (
            <div className="mt-3 rounded-md border border-primary/15 bg-primary/[0.035] px-3 py-2 text-[10.5px] text-muted-foreground">
              Training complete. Artifacts written to the project model folder.
            </div>
          ) : null}

          <StreamingTest
            evaluation={summary.model.evaluation}
            modelExists={summary.model.exists}
            corpora={corpora}
            jobBusy={jobBusy}
            jobKind={jobKind}
            progress={progress}
            jobError={jobError}
            interactionDisabled={interactionDisabled}
            onStart={onStartEvaluation}
            onCancel={onCancelJob}
            onOpenSpeechData={onOpenSpeechData}
            minedCount={summary.minedCount}
            canMine={corpora.some((corpus) => corpus.role === "negatives" && corpus.installed)}
            onMine={onStartMining}
          />

          {jobBusy && jobKind === "augment" ? (
            <div className="mt-3 flex items-center gap-2 rounded-md border border-amber-400/15 bg-amber-400/[0.035] px-3 py-2 text-[10.5px] text-amber-100/70">
              <Loader2 className="h-3 w-3 animate-spin" />
              Dataset preparation is running. Training can start when it finishes.
            </div>
          ) : null}
        </div>
      </div>

      <div className="px-5 pb-5">
        <LogConsole logs={logs} busy={jobBusy} onClear={onClearLogs} maxHeight={210} />
      </div>
    </section>
  );
}
