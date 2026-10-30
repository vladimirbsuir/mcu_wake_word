import { AudioLines, Loader2, Pickaxe, Radio, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { PipelineKind } from "@/hooks/useJobRunner";
import { cn, formatDate } from "@/lib/utils";
import type { SpeechCorpus } from "@/lib/voice";
import type { JobProgress, StreamingEvaluation } from "@/types";

const TABLE_THRESHOLDS = ["0.50", "0.80", "0.90", "0.95", "0.98", "0.99"];

type StreamingTestProps = {
  evaluation: StreamingEvaluation | null;
  modelExists: boolean;
  corpora: SpeechCorpus[];
  jobBusy: boolean;
  jobKind: PipelineKind;
  progress: JobProgress | null;
  jobError: string | null;
  interactionDisabled: boolean;
  onStart: () => void;
  onCancel: () => void;
  onOpenSpeechData: () => void;
  minedCount: number;
  canMine: boolean;
  onMine: () => void;
};

function formatRate(value: number): string {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

/** "all:3" → "3 in a row", "mean:4" → "avg of 4". */
function ruleLabel(detector: string): string {
  const [mode, windows] = detector.split(":");
  return mode === "mean" ? `avg of ${windows}` : `${windows} in a row`;
}

function formatRecall(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

/**
 * Firmware-style benchmark: hours of unseen speech streamed through the INT8
 * model. Shows false activations per hour against recall and the detector
 * settings exported in wake_word_detector.h.
 */
export function StreamingTest({
  evaluation,
  modelExists,
  corpora,
  jobBusy,
  jobKind,
  progress,
  jobError,
  interactionDisabled,
  onStart,
  onCancel,
  onOpenSpeechData,
  minedCount,
  canMine,
  onMine,
}: StreamingTestProps) {
  const benchmarks = corpora.filter((corpus) => corpus.role === "eval" && corpus.installed);
  const running = jobBusy && (jobKind === "evaluate" || jobKind === "mine");
  const chosen = evaluation?.detector ?? evaluation?.operatingPoints?.[0] ?? null;
  const error = jobKind === "evaluate" || jobKind === "mine" ? jobError : null;

  return (
    <div className="mt-3 rounded-lg border border-border/70 bg-background/30 p-3.5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 label-caps">
            <Radio className="h-3 w-3" /> Streaming test
          </div>
          <p className="mt-1 max-w-lg text-[10.5px] leading-relaxed text-zinc-600">
            Unseen speech is streamed through the INT8 model every {evaluation?.stepMs ?? 100} ms like on
            the ESP32. Accuracy on clips says little; false activations per hour is what you will feel.
          </p>
        </div>
        {running ? (
          <Button variant="ghost" size="sm" className="h-7 text-[10px]" onClick={onCancel}>
            <Square className="h-2.5 w-2.5" /> Stop
          </Button>
        ) : benchmarks.length ? (
          <Button
            variant="outline"
            size="sm"
            onClick={onStart}
            disabled={!modelExists || jobBusy || interactionDisabled}
            title={!modelExists ? "Train a model first" : undefined}
          >
            <Radio className="h-3 w-3" /> {evaluation ? "Run again" : "Run test"}
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={onOpenSpeechData}>
            <AudioLines className="h-3 w-3" /> Get benchmark speech
          </Button>
        )}
      </div>

      {running ? (
        <div className="mt-3 space-y-1.5">
          <div className="flex items-center gap-2 text-[10.5px] text-cyan-100/80">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span className="truncate">{progress?.message ?? "Starting…"}</span>
          </div>
          <Progress value={(progress?.progress ?? 0) * 100} className="h-1" indicatorClassName="bg-cyan-300" />
        </div>
      ) : null}

      {error ? <p className="mt-2 text-[11px] text-red-300">{error}</p> : null}

      {evaluation && !running ? (
        <>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { label: "False / hour", value: chosen ? formatRate(chosen.faPerHour) : "—", accent: true },
              { label: "Recall", value: formatRecall(chosen?.recall), accent: true },
              { label: "Threshold", value: chosen ? chosen.threshold.toFixed(2) : "—" },
              { label: "Rule", value: chosen ? ruleLabel(chosen.detector) : "—" },
            ].map((tile) => (
              <div key={tile.label} className="rounded-md border border-border/60 bg-background/40 px-3 py-2">
                <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-600">{tile.label}</div>
                <div className={cn("mt-1 font-mono text-lg leading-none", tile.accent ? "text-primary" : "text-zinc-200")}>
                  {tile.value}
                </div>
              </div>
            ))}
          </div>
          {!chosen ? (
            <p className="mt-2 text-[10.5px] text-amber-200/70">
              No setting reaches 2 false activations per hour. Add more real speech or your own
              other-speech takes and retrain.
            </p>
          ) : null}

          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[420px] font-mono text-[10px] text-zinc-500">
              <thead>
                <tr className="text-zinc-600">
                  <th className="py-1 pr-2 text-left font-normal">rule \ threshold</th>
                  {TABLE_THRESHOLDS.map((threshold) => (
                    <th key={threshold} className="px-1.5 py-1 text-right font-normal">
                      {threshold}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(evaluation.curve).map(([detector, row]) => (
                  <tr key={detector} className="border-t border-border/40">
                    <td className="whitespace-nowrap py-1 pr-2">{ruleLabel(detector)}</td>
                    {TABLE_THRESHOLDS.map((threshold) => {
                      const cell = row[threshold];
                      const highlight =
                        chosen &&
                        chosen.detector === detector &&
                        chosen.threshold.toFixed(2) === threshold;
                      return (
                        <td
                          key={threshold}
                          className={cn("whitespace-nowrap px-1.5 py-1 text-right", highlight && "text-primary")}
                          title="false activations per hour · recall"
                        >
                          {cell ? `${formatRate(cell.faPerHour)} · ${formatRecall(cell.recall)}` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-700">
            <span>{evaluation.speechHours.toFixed(2)} h of audio</span>
            <span>{evaluation.recall.trials ?? 0} recall trials · {evaluation.recall.takes?.length ?? 0} held-out takes</span>
            <span>{formatDate(evaluation.createdAt)}</span>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border border-border/50 bg-background/30 px-3 py-2 text-[10.5px] text-zinc-500">
            <span className="min-w-0">
              Mine hard negatives: scan the training speech for windows this model fires on and add
              them to the next dataset. Then rebuild, retrain and test again.
              {minedCount ? ` ${minedCount} mined so far.` : ""}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="h-7"
              onClick={onMine}
              disabled={!canMine || jobBusy || interactionDisabled}
              title={canMine ? undefined : "Download a training speech set first"}
            >
              <Pickaxe className="h-3 w-3" /> Mine
            </Button>
          </div>

          {evaluation.perTake?.length ? (
            <details className="mt-2 text-[10.5px] text-zinc-500" open={evaluation.perTake.some((row) => row.recall < 0.7)}>
              <summary className="cursor-pointer text-zinc-600 hover:text-zinc-400">
                Recall by held-out take
                {evaluation.perTake.some((row) => row.recall < 0.7)
                  ? ` · ${evaluation.perTake.filter((row) => row.recall < 0.7).length} weak`
                  : ""}
              </summary>
              <table className="mt-1.5 w-full font-mono text-[10px]">
                <tbody>
                  {evaluation.perTake.map((row) => (
                    <tr key={row.take} className="border-t border-border/30">
                      <td className="py-0.5 pr-2 text-zinc-400">{row.take.replace(/\.wav$/, "")}</td>
                      <td className={cn("px-1.5 text-right", row.recall < 0.7 ? "text-amber-300" : "text-zinc-300")}>
                        {formatRecall(row.recall)}
                      </td>
                      <td className="px-1.5 text-right text-zinc-600" title="median best score around the take">
                        peak {row.medianPeak.toFixed(2)}
                      </td>
                      <td className="pl-1.5 text-zinc-600">
                        {Object.entries(row.byContext)
                          .map(([context, value]) => `${context} ${value}`)
                          .join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1 text-zinc-600">
                Weak takes are worth listening to: a cut-off, mumbled or very quiet take teaches the model
                the wrong thing. Re-record or delete it and retrain.
              </p>
            </details>
          ) : null}

          {evaluation.falseActivations.some((item) => item.top.length) ? (
            <details className="mt-2 text-[10.5px] text-zinc-500">
              <summary className="cursor-pointer text-zinc-600 hover:text-zinc-400">
                What it confuses with the phrase
              </summary>
              <ul className="mt-1.5 space-y-0.5">
                {evaluation.falseActivations
                  .flatMap((item) => item.top)
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 10)
                  .map((item, index) => (
                    <li key={index} className="flex gap-2">
                      <span className="w-10 shrink-0 font-mono text-zinc-600">{item.score.toFixed(2)}</span>
                      <span className="truncate">{item.text || "—"}</span>
                    </li>
                  ))}
              </ul>
            </details>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
