import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "@/lib/api";
import type { CaptureStart, Meter, StopRecordingResult, TakeKind } from "@/types";

export const AUTO_STOP_MS = 10_000;

type RecorderPhase = "idle" | "starting" | "recording" | "stopping";

const EMPTY_METER: Meter = { levels: [], rms: 0, peak: 0, durationMs: 0 };

export function useRecorder(
  project: string,
  disabled: boolean,
  onSaved: () => void | Promise<void>,
) {
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [capture, setCapture] = useState<CaptureStart | null>(null);
  const [meter, setMeter] = useState<Meter>(EMPTY_METER);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<StopRecordingResult | null>(null);
  const phaseRef = useRef<RecorderPhase>("idle");
  const projectRef = useRef(project);
  const onSavedRef = useRef(onSaved);
  const mountedRef = useRef(true);
  const pollRef = useRef<number | null>(null);
  const stopRef = useRef<() => Promise<void>>(async () => undefined);

  useEffect(() => {
    projectRef.current = project;
  }, [project]);

  useEffect(() => {
    onSavedRef.current = onSaved;
  }, [onSaved]);

  const clearPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const setPhaseBoth = useCallback((next: RecorderPhase) => {
    phaseRef.current = next;
    if (mountedRef.current) setPhase(next);
  }, []);

  const stop = useCallback(async () => {
    if (phaseRef.current !== "recording") return;
    setPhaseBoth("stopping");
    clearPolling();
    try {
      // The backend saves into the project the capture was started for.
      const result = await api.stopRecording();
      if (!mountedRef.current) return;
      setLastResult(result);
      setError(null);
      setMeter(EMPTY_METER);
      await onSavedRef.current();
    } catch (stopError) {
      if (mountedRef.current) setError(errorMessage(stopError));
    } finally {
      if (mountedRef.current) {
        setCapture(null);
        setMeter(EMPTY_METER);
      }
      setPhaseBoth("idle");
    }
  }, [clearPolling, setPhaseBoth]);

  stopRef.current = stop;

  const startPolling = useCallback(() => {
    clearPolling();
    let inFlight = false;
    pollRef.current = window.setInterval(async () => {
      if (inFlight || phaseRef.current !== "recording") return;
      inFlight = true;
      try {
        const nextMeter = await api.getRecordingMeter();
        if (!mountedRef.current || phaseRef.current !== "recording") return;
        setMeter(nextMeter);
        if (nextMeter.durationMs >= AUTO_STOP_MS) {
          void stopRef.current();
        }
      } catch {
        // A transient meter failure should not kill an otherwise valid take.
      } finally {
        inFlight = false;
      }
    }, 90);
  }, [clearPolling]);

  const start = useCallback(async (kind: TakeKind = "wake", prompt: string | null = null) => {
    if (disabled || phaseRef.current !== "idle") return;
    setPhaseBoth("starting");
    setError(null);
    setLastResult(null);
    setMeter(EMPTY_METER);
    try {
      const info = await api.startRecording(projectRef.current, kind, prompt);
      if (!mountedRef.current) {
        void api.cancelRecording().catch(() => undefined);
        return;
      }
      setCapture(info);
      setPhaseBoth("recording");
      startPolling();
    } catch (startError) {
      if (mountedRef.current) {
        setError(errorMessage(startError));
        setCapture(null);
      }
      setPhaseBoth("idle");
    }
  }, [disabled, setPhaseBoth, startPolling]);

  const discard = useCallback(async () => {
    if (phaseRef.current !== "recording" && phaseRef.current !== "starting") return;
    clearPolling();
    setPhaseBoth("stopping");
    try {
      await api.cancelRecording();
    } catch (cancelError) {
      if (mountedRef.current) setError(errorMessage(cancelError));
    } finally {
      if (mountedRef.current) {
        setCapture(null);
        setMeter(EMPTY_METER);
        setPhaseBoth("idle");
      }
    }
  }, [clearPolling, setPhaseBoth]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearPolling();
      if (phaseRef.current === "recording" || phaseRef.current === "starting") {
        void api.cancelRecording().catch(() => undefined);
      }
    };
  }, [clearPolling]);

  return {
    phase,
    capture,
    meter,
    error,
    lastResult,
    isRecording: phase === "recording",
    isBusy: phase === "starting" || phase === "stopping",
    start,
    stop,
    discard,
    clearError: () => setError(null),
    autoStopMs: AUTO_STOP_MS,
  };
}
