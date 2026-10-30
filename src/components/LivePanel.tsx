import { useEffect, useRef, useState } from "react";
import { Check, Ear, Loader2, Mic, Pause, Play, Radio, Square } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type { PipelineKind } from "@/hooks/useJobRunner";
import { api, errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { JobFinishedEvent, JobProgress, LiveEvent, LiveParams, LiveProgress, ProjectSummary } from "@/types";

const HISTORY_POINTS = 100; // 10 s of 100 ms windows
// Peak level range the model was trained on (python/augment.py).
const TRAINED_PEAK_DB: [number, number] = [-44, -8];

type LivePanelProps = {
  summary: ProjectSummary;
  environmentReady: boolean;
  jobBusy: boolean;
  jobKind: PipelineKind;
  jobProject: string | null;
  progress: JobProgress | null;
  finished: JobFinishedEvent | null;
  jobError: string | null;
  interactionDisabled: boolean;
  onStart: (params: LiveParams) => void;
  onStop: () => void;
  onRefresh: () => void | Promise<void>;
};

function defaults(summary: ProjectSummary): Omit<LiveParams, "project"> {
  const detector = summary.model.evaluation?.detector;
  return {
    rule: detector?.mode ?? "all",
    windows: detector?.windows ?? 3,
    threshold: detector?.threshold ?? 0.9,
    refractoryMs: 1500,
    gainDb: 0,
  };
}

function Meter({ value, min, max, marks, className }: { value: number; min: number; max: number; marks?: [number, number]; className?: string }) {
  const position = (x: number) => `${Math.max(0, Math.min(100, ((x - min) / (max - min)) * 100))}%`;
  return (
    <div className={cn("relative h-2 overflow-hidden rounded-full bg-secondary", className)}>
      {marks ? (
        <div
          className="absolute inset-y-0 bg-primary/10"
          style={{ left: position(marks[0]), width: `calc(${position(marks[1])} - ${position(marks[0])})` }}
        />
      ) : null}
      <div className="absolute inset-y-0 left-0 rounded-full bg-cyan-300/70 transition-[width] duration-100" style={{ width: position(value) }} />
    </div>
  );
}

function Sparkline({ scores, threshold, detections }: { scores: number[]; threshold: number; detections: number[] }) {
  const width = 600;
  const height = 90;
  const x = (index: number) => (index / (HISTORY_POINTS - 1)) * width;
  const y = (value: number) => height - value * (height - 6) - 3;
  const offset = HISTORY_POINTS - scores.length;
  const path = scores.map((value, index) => `${index ? "L" : "M"}${x(index + offset).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[90px] w-full" preserveAspectRatio="none" role="img" aria-label="Score over the last 10 seconds">
      <line x1={0} x2={width} y1={y(threshold)} y2={y(threshold)} stroke="#fbbf24" strokeOpacity={0.5} strokeDasharray="4 4" />
      {detections.map((index) => (
        <line key={index} x1={x(index + offset)} x2={x(index + offset)} y1={0} y2={height} stroke="#c8f560" strokeOpacity={0.5} />
      ))}
      <path d={path} fill="none" stroke="#67e8f9" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/**
 * Step 06: the trained model on the PC microphone in real time, with the same
 * detector as the streaming test and the ESP32 firmware. Detections and
 * "missed" snapshots can be played back and kept as training data.
 */
export function LivePanel({
  summary,
  environmentReady,
  jobBusy,
  jobKind,
  jobProject,
  progress,
  finished,
  jobError,
  interactionDisabled,
  onStart,
  onStop,
  onRefresh,
}: LivePanelProps) {
  const [params, setParams] = useState(() => defaults(summary));
  const [history, setHistory] = useState<{ ms: number; scores: number[]; detections: number[] }>({ ms: -1, scores: [], detections: [] });
  const [saved, setSaved] = useState<Record<string, "wake" | "negative">>({});
  const [playing, setPlaying] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const running = jobBusy && jobKind === "live" && jobProject === summary.slug;
  const live = running && progress?.kind === "live" ? (progress as unknown as LiveProgress) : null;
  const lastLive = useRef<LiveProgress | null>(null);
  const lastDetections = useRef(0);
  if (live && typeof live.score === "number") lastLive.current = live;
  const shown = live ?? (finished?.kind === "live" ? lastLive.current : null);
  const events: LiveEvent[] = shown?.events ?? [];

  // Rolling score history, one point per window.
  useEffect(() => {
    if (!live || typeof live.score !== "number" || live.ms === history.ms) return;
    setHistory((current) => {
      const scores = [...current.scores, live.score].slice(-HISTORY_POINTS);
      const shift = current.scores.length + 1 - scores.length;
      const detections = current.detections.map((index) => index - shift).filter((index) => index >= 0);
      const known = current.scores.length ? lastDetections.current : live.detections;
      if (live.detections > known) detections.push(scores.length - 1);
      lastDetections.current = live.detections;
      return { ms: live.ms, scores, detections };
    });
  }, [live, history.ms]);

  useEffect(() => () => audioRef.current?.pause(), []);

  const start = () => {
    setHistory({ ms: -1, scores: [], detections: [] });
    lastDetections.current = 0;
    setSaved({});
    setActionError(null);
    onStart({ project: summary.slug, ...params });
  };

  const play = async (file: string) => {
    audioRef.current?.pause();
    if (playing === file) {
      setPlaying(null);
      return;
    }
    try {
      const bytes = await api.readLiveClip(summary.slug, file);
      const url = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: "audio/wav" }));
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        setPlaying(null);
        URL.revokeObjectURL(url);
      };
      setPlaying(file);
      await audio.play();
    } catch (error) {
      setActionError(errorMessage(error));
      setPlaying(null);
    }
  };

  const keep = async (file: string, kind: "wake" | "negative") => {
    setActionError(null);
    try {
      await api.saveLiveClip(summary.slug, file, kind);
      setSaved((current) => ({ ...current, [file]: kind }));
      await onRefresh();
    } catch (error) {
      setActionError(errorMessage(error));
    }
  };

  const error = actionError ?? (finished?.kind === "live" && !finished.success && !finished.canceled ? jobError ?? finished.message : null);
  const tested = summary.model.evaluation?.detector;
  const disabledControls = running || jobBusy || interactionDisabled;

  return (
    <section id="live" className="panel scroll-mt-24 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">06</span>
            <span className="label-caps">Live test</span>
          </div>
          <h2 className="mt-2 text-lg font-semibold tracking-tight">Try it on your microphone</h2>
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted-foreground">
            The INT8 model listens to the PC microphone ten times a second with the same detector as
            the streaming test and the ESP32 firmware. Say the phrase, talk around it, play a video:
            every detection is saved so you can listen to it and keep it as training data.
          </p>
        </div>
        {running ? (
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => void api.liveSnapshot(summary.slug).catch((e) => setActionError(errorMessage(e)))}>
              <Ear className="h-3.5 w-3.5" /> It missed
            </Button>
            <Button variant="destructive" size="sm" onClick={onStop}>
              <Square className="h-3 w-3 fill-current" /> Stop
            </Button>
          </div>
        ) : (
          <Button
            size="sm"
            onClick={start}
            disabled={!summary.model.exists || !environmentReady || jobBusy || interactionDisabled}
            title={!summary.model.exists ? "Train a model first" : jobBusy ? "Another job is running" : undefined}
          >
            <Mic className="h-3.5 w-3.5" /> Start listening
          </Button>
        )}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[250px_minmax(0,1fr)]">
        <div className="space-y-3 rounded-lg border border-border/70 bg-background/30 p-3.5">
          <div className="label-caps">Detector</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <Label htmlFor="live-rule">Rule</Label>
              <select
                id="live-rule"
                value={params.rule}
                onChange={(event) => setParams({ ...params, rule: event.target.value as "all" | "mean" })}
                disabled={disabledControls}
                className="focus-ring flex h-8 w-full rounded-md border border-input bg-background/60 px-2 text-xs disabled:opacity-50"
              >
                <option value="all">in a row</option>
                <option value="mean">average</option>
              </select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="live-windows">Windows</Label>
              <select
                id="live-windows"
                value={params.windows}
                onChange={(event) => setParams({ ...params, windows: Number(event.target.value) })}
                disabled={disabledControls}
                className="focus-ring flex h-8 w-full rounded-md border border-input bg-background/60 px-2 font-mono text-xs disabled:opacity-50"
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <Label htmlFor="live-threshold">Threshold</Label>
              <span className="font-mono text-[11px] text-zinc-300">{params.threshold.toFixed(2)}</span>
            </div>
            <input
              id="live-threshold"
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={params.threshold}
              onChange={(event) => setParams({ ...params, threshold: Number(event.target.value) })}
              disabled={disabledControls}
              className="h-1 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-[#c8f560] disabled:opacity-40"
            />
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <Label htmlFor="live-gain">Input gain</Label>
              <span className="font-mono text-[11px] text-zinc-300">{params.gainDb > 0 ? "+" : ""}{params.gainDb} dB</span>
            </div>
            <input
              id="live-gain"
              type="range"
              min={-12}
              max={24}
              step={3}
              value={params.gainDb}
              onChange={(event) => setParams({ ...params, gainDb: Number(event.target.value) })}
              disabled={disabledControls}
              className="h-1 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-[#c8f560] disabled:opacity-40"
            />
          </div>
          <p className="text-[10px] leading-relaxed text-zinc-600">
            {tested
              ? `Streaming test picked ${tested.mode === "mean" ? "average" : "in a row"} of ${tested.windows} at ${tested.threshold.toFixed(2)} (${tested.faPerHour} false/h). `
              : "Run the streaming test to get a measured setting. "}
            Settings apply on the next start.
          </p>
          {tested && (params.rule !== tested.mode || params.windows !== tested.windows || params.threshold !== tested.threshold) ? (
            <button
              type="button"
              onClick={() => setParams({ ...params, rule: tested.mode, windows: tested.windows, threshold: tested.threshold })}
              disabled={disabledControls}
              className="text-[10.5px] text-cyan-300/80 hover:text-cyan-200 disabled:opacity-40"
            >
              Use the tested setting
            </button>
          ) : null}
        </div>

        <div className="min-w-0 space-y-3">
          <div className="rounded-lg border border-border/70 bg-[#080b09] p-3.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
                <span className={cn("h-1.5 w-1.5 rounded-full", running ? "animate-blink bg-red-400" : "bg-zinc-600")} />
                {running ? (shown?.device ? `listening · ${shown.device}` : "starting…") : "stopped"}
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={shown?.detections ? "default" : "outline"}>
                  <Radio className="h-2.5 w-2.5" /> {shown?.detections ?? 0} detections
                </Badge>
              </div>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-[120px_minmax(0,1fr)]">
              <div>
                <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-600">score</div>
                <div
                  className={cn(
                    "font-mono text-4xl leading-none tabular-nums",
                    shown && shown.rule >= params.threshold ? "text-primary" : "text-zinc-300",
                  )}
                >
                  {shown ? shown.score.toFixed(2) : "—"}
                </div>
                <div className="mt-1 font-mono text-[10px] text-zinc-600">
                  rule {shown ? shown.rule.toFixed(2) : "—"} / {params.threshold.toFixed(2)}
                </div>
              </div>
              <Sparkline scores={history.scores} threshold={params.threshold} detections={history.detections} />
            </div>
            <div className="mt-3 space-y-1">
              <div className="flex justify-between font-mono text-[9.5px] text-zinc-600">
                <span>input peak {shown ? `${shown.peakDb.toFixed(1)} dBFS` : "—"} · rms {shown ? shown.rmsDb.toFixed(1) : "—"}</span>
                <span>trained range {TRAINED_PEAK_DB[0]}…{TRAINED_PEAK_DB[1]} dBFS</span>
              </div>
              <Meter value={shown?.peakDb ?? -90} min={-70} max={0} marks={TRAINED_PEAK_DB} />
              {running && shown && shown.peakDb > -1 ? (
                <p className="text-[10.5px] text-amber-300/80">The input is clipping: lower the gain.</p>
              ) : null}
            </div>
          </div>

          {error ? <p className="text-[11px] text-red-300">{error}</p> : null}

          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="label-caps">Detections and misses</span>
              <span className="text-[10px] text-zinc-600">kept clips go to Capture and rebuild into the next dataset</span>
            </div>
            {events.length === 0 ? (
              <p className="rounded-md border border-dashed border-border/60 px-3 py-3 text-[11px] text-zinc-600">
                {running
                  ? "Say the phrase. If it does not react, press “It missed” right after to save the last 4 s."
                  : "Nothing yet. Start listening to see detections here."}
              </p>
            ) : (
              <div className="max-h-[230px] space-y-1 overflow-y-auto pr-1">
                {events.map((event) => {
                  const kept = saved[event.file];
                  return (
                    <div key={event.file} className="flex flex-wrap items-center gap-2 rounded-md border border-border/50 bg-background/25 px-2.5 py-1.5 text-[11px]">
                      <button
                        type="button"
                        onClick={() => void play(event.file)}
                        disabled={!event.saved}
                        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-primary disabled:opacity-40"
                        aria-label="Play clip"
                      >
                        {!event.saved ? <Loader2 className="h-3 w-3 animate-spin" /> : playing === event.file ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3 pl-0.5" />}
                      </button>
                      <span className={cn("font-mono text-[10px] uppercase", event.kind === "missed" ? "text-amber-300/80" : "text-primary/80")}>
                        {event.kind}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-500">{event.at}</span>
                      {event.kind === "detection" ? (
                        <span className="font-mono text-[10px] text-zinc-500">
                          score {event.score?.toFixed(2)} · peak {event.peakDb?.toFixed(0)} dBFS
                        </span>
                      ) : null}
                      <span className="flex-1" />
                      {kept ? (
                        <span className="flex items-center gap-1 text-[10.5px] text-zinc-400">
                          <Check className="h-3 w-3 text-primary" /> kept as {kept === "wake" ? "wake take" : "other speech"}
                        </span>
                      ) : (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-[10px]"
                            disabled={!event.saved}
                            onClick={() => void keep(event.file, "wake")}
                            title="It was the wake phrase: add the clip to the wake takes"
                          >
                            Wake take
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-[10px]"
                            disabled={!event.saved}
                            onClick={() => void keep(event.file, "negative")}
                            title="It was not the phrase: keep it as other speech that must not trigger"
                          >
                            Other speech
                          </Button>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
