import { useCallback, useEffect, useRef, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api, errorMessage } from "@/lib/api";
import type {
  EpochMetrics,
  JobFinishedEvent,
  JobLogEvent,
  JobProgress,
  JobStarted,
  ModelArchitecture,
} from "@/types";

export type PipelineKind =
  | "augment"
  | "train"
  | "install"
  | "packages"
  | "check"
  | "transcribe"
  | "words"
  | "synthesize"
  | "corpus"
  | "evaluate"
  | "mine"
  | "live"
  | null;

/** Human-readable name of a job kind, for banners and lock messages. */
export const JOB_TITLES: Record<Exclude<PipelineKind, null>, string> = {
  augment: "Dataset",
  train: "Training",
  install: "Model download",
  packages: "Python packages",
  check: "Provider test",
  transcribe: "Speech recognition",
  words: "Word lists",
  synthesize: "Voice synthesis",
  corpus: "Speech data download",
  evaluate: "Streaming test",
  mine: "Hard negative mining",
  live: "Live test",
};

export type LogLine = {
  id: number;
  stream: string;
  line: string;
  timestamp: string;
};

type StartState = {
  busy: boolean;
  kind: PipelineKind;
  /** Project the current (or last) job belongs to. */
  project: string | null;
  started: JobStarted | null;
  progress: JobProgress | null;
  logs: LogLine[];
  liveHistory: EpochMetrics[];
  finished: JobFinishedEvent | null;
  error: string | null;
};

const INITIAL: StartState = {
  busy: false,
  kind: null,
  project: null,
  started: null,
  progress: null,
  logs: [],
  liveHistory: [],
  finished: null,
  error: null,
};

const MAX_LOG_LINES = 900;

export function useJobRunner(onFinished: () => void | Promise<void>) {
  const [state, setState] = useState<StartState>(INITIAL);
  const busyRef = useRef(false);
  // Events carry the project they belong to; anything else is ignored.
  const projectRef = useRef<string | null>(null);
  const logIdRef = useRef(0);
  const onFinishedRef = useRef(onFinished);
  const mountedRef = useRef(true);

  useEffect(() => {
    onFinishedRef.current = onFinished;
  }, [onFinished]);

  useEffect(() => {
    mountedRef.current = true;
    let unlistenLog: UnlistenFn | undefined;
    let unlistenProgress: UnlistenFn | undefined;
    let unlistenFinished: UnlistenFn | undefined;
    let cancelled = false;

    const setup = async () => {
      unlistenLog = await listen<JobLogEvent>("job-log", (event) => {
        if (!busyRef.current || !mountedRef.current) return;
        if (event.payload.project !== projectRef.current) return;
        const entry: LogLine = {
          id: ++logIdRef.current,
          stream: event.payload.stream,
          line: event.payload.line,
          timestamp: event.payload.timestamp,
        };
        setState((previous) => ({
          ...previous,
          logs: [...previous.logs.slice(-(MAX_LOG_LINES - 1)), entry],
        }));
      });

      unlistenProgress = await listen<JobProgress>("job-progress", (event) => {
        if (!busyRef.current || !mountedRef.current) return;
        if (event.payload.project !== projectRef.current) return;
        const progress = event.payload;
        setState((previous) => {
          const next = { ...previous, progress };
          if (progress.metrics && typeof progress.metrics.epoch === "number") {
            const byEpoch = new Map(previous.liveHistory.map((item) => [item.epoch, item]));
            byEpoch.set(progress.metrics.epoch, progress.metrics);
            next.liveHistory = Array.from(byEpoch.values()).sort((a, b) => a.epoch - b.epoch);
          }
          return next;
        });
      });

      unlistenFinished = await listen<JobFinishedEvent>("job-finished", (event) => {
        if (!mountedRef.current) return;
        if (!busyRef.current) return;
        if (event.payload.project !== projectRef.current) return;
        busyRef.current = false;
        setState((previous) => ({
          ...previous,
          busy: false,
          finished: event.payload,
          error: event.payload.success ? null : event.payload.message,
          logs: event.payload.success
            ? previous.logs
            : [
                ...previous.logs.slice(-(MAX_LOG_LINES - 1)),
                {
                  id: ++logIdRef.current,
                  stream: "stderr",
                  line: event.payload.message,
                  timestamp: new Date().toISOString(),
                },
              ],
        }));
        void onFinishedRef.current();
      });

      if (cancelled) {
        unlistenLog?.();
        unlistenProgress?.();
        unlistenFinished?.();
      }
    };

    void setup().catch((setupError) => {
      if (mountedRef.current) {
        setState((previous) => ({
          ...previous,
          error: `Cannot subscribe to pipeline events: ${errorMessage(setupError)}`,
        }));
      }
    });

    return () => {
      cancelled = true;
      unlistenLog?.();
      unlistenProgress?.();
      unlistenFinished?.();
    };
  }, []);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const begin = useCallback(
    async (
      kind: Exclude<PipelineKind, null>,
      project: string,
      starter: () => Promise<JobStarted>,
    ) => {
      if (busyRef.current) return;
      busyRef.current = true;
      projectRef.current = project;
      setState({
        ...INITIAL,
        busy: true,
        kind,
        project,
      });
      try {
        const started = await starter();
        if (!mountedRef.current) return;
        setState((previous) => ({ ...previous, started }));
      } catch (startError) {
        busyRef.current = false;
        if (!mountedRef.current) return;
        setState((previous) => ({
          ...previous,
          busy: false,
          error: errorMessage(startError),
          logs: [
            {
              id: ++logIdRef.current,
              stream: "stderr",
              line: errorMessage(startError),
              timestamp: new Date().toISOString(),
            },
          ],
        }));
      }
    },
    [],
  );

  const startAugmentation = useCallback(
    (params: {
      project: string;
      variations: number;
      clipSeconds: number;
      backgroundDir: string | null;
      negativeDir: string | null;
      useSynthetic: boolean;
      useCorpora: boolean;
      corpusNegatives: number;
      useMined: boolean;
    }) => begin("augment", params.project, () => api.startAugmentation(params)),
    [begin],
  );

  const startTraining = useCallback(
    (params: {
      project: string;
      epochs: number;
      batchSize: number;
      learningRate: number;
      architecture: ModelArchitecture;
    }) => begin("train", params.project, () => api.startTraining(params)),
    [begin],
  );

  const cancel = useCallback(async () => {
    if (!busyRef.current) return;
    try {
      await api.cancelJob();
      setState((previous) => ({
        ...previous,
        progress: previous.progress
          ? { ...previous.progress, message: "Cancelling…" }
          : previous.progress,
      }));
    } catch (cancelError) {
      setState((previous) => ({ ...previous, error: errorMessage(cancelError) }));
    }
  }, []);

  const clearLogs = useCallback(() => {
    setState((previous) => ({ ...previous, logs: [] }));
  }, []);

  /** Start any job kind; `project` is "" for global jobs such as downloads. */
  const run = useCallback(
    (kind: Exclude<PipelineKind, null>, project: string, starter: () => Promise<JobStarted>) =>
      begin(kind, project, starter),
    [begin],
  );

  /** Forget the last job's logs and results, e.g. after switching projects. */
  const reset = useCallback(() => {
    if (busyRef.current) return;
    projectRef.current = null;
    setState(INITIAL);
  }, []);

  return {
    ...state,
    startAugmentation,
    startTraining,
    cancel,
    clearLogs,
    reset,
    run,
  };
}
