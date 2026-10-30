import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  RefreshCw,
  Terminal,
  X,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { EnvironmentStatus } from "@/types";

type EnvironmentDialogProps = {
  open: boolean;
  environment: EnvironmentStatus | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRefresh: () => void;
};

const PACKAGE_ORDER = ["tensorflow", "librosa", "resampy", "soundfile", "numpy"];
const SETUP_WINDOWS = `py -3.12 -m venv .venv
.venv\\Scripts\\python -m pip install --upgrade pip
.venv\\Scripts\\pip install -r python\\requirements.txt`;
const SETUP_UNIX = `python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r python/requirements.txt`;

export function EnvironmentDialog({
  open,
  environment,
  loading,
  error,
  onClose,
  onRefresh,
}: EnvironmentDialogProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  useEffect(() => {
    setCopied(false);
  }, [open]);

  if (!open) return null;

  const copySetup = async () => {
    try {
      await navigator.clipboard.writeText(SETUP_WINDOWS);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Ignore clipboard denial.
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm animate-fade-in"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        className="panel max-h-[88vh] w-full max-w-2xl overflow-y-auto bg-[#0c100e] shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="environment-title"
      >
        <div className="flex items-start justify-between border-b border-border/70 px-6 py-5">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-secondary/60">
              <Terminal className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h2 id="environment-title" className="text-base font-semibold">
                ML environment
              </h2>
              <p className="mt-1 text-xs text-muted-foreground">
                WakeForge runs the augmentation and TensorFlow pipeline through a local Python
                process.
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close dialog">
            <X />
          </Button>
        </div>

        <div className="space-y-5 p-6">
          <div
            className={cn(
              "flex items-start gap-3 rounded-lg border p-3.5",
              environment?.ready
                ? "border-primary/20 bg-primary/[0.055]"
                : "border-amber-400/20 bg-amber-400/[0.045]",
            )}
          >
            {environment?.ready ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            ) : (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
            )}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium">
                  {loading
                    ? "Checking environment…"
                    : environment?.ready
                      ? "Ready to train"
                      : "Setup required"}
                </span>
                {environment?.version ? (
                  <Badge variant={environment.supportedPython ? "default" : "amber"}>
                    Python {environment.version}
                  </Badge>
                ) : null}
                {environment?.ready ? <Badge variant="cyan">TensorFlow</Badge> : null}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {loading ? "Probing the interpreter and installed packages." : error || environment?.message}
              </p>
              {environment?.python ? (
                <p className="mt-2 truncate font-mono text-[10px] text-muted-foreground/60">
                  {environment.python}
                </p>
              ) : null}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              disabled={loading}
              className="shrink-0"
            >
              <RefreshCw className={cn(loading && "animate-spin")} />
              Recheck
            </Button>
          </div>

          {environment ? (
            <div>
              <div className="label-caps mb-2.5">Python packages</div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {PACKAGE_ORDER.map((name) => {
                  const installed = environment.packages[name];
                  return (
                    <div
                      key={name}
                      className="flex items-center justify-between gap-2 rounded-md border border-border/70 bg-background/40 px-2.5 py-2"
                    >
                      <span className="font-mono text-[11px] text-zinc-300">{name}</span>
                      {installed ? (
                        <Check className="h-3.5 w-3.5 text-primary" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-red-400/80" />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          <div>
            <div className="mb-2.5 flex items-center justify-between">
              <div className="label-caps">First-time setup · Windows</div>
              <button
                type="button"
                onClick={copySetup}
                className="font-mono text-[10px] text-muted-foreground transition-colors hover:text-primary"
              >
                {copied ? "COPIED" : "COPY"}
              </button>
            </div>
            <pre className="overflow-x-auto rounded-lg border border-border/70 bg-[#070a08] p-3.5 font-mono text-[11px] leading-relaxed text-zinc-400">
              {SETUP_WINDOWS}
            </pre>
            <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
              macOS / Linux:
            </p>
            <pre className="mt-1 overflow-x-auto rounded-lg border border-border/70 bg-[#070a08] p-3.5 font-mono text-[11px] leading-relaxed text-zinc-400">
              {SETUP_UNIX}
            </pre>
            <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
              The app automatically prefers <code className="text-zinc-300">.venv</code> in the
              project root. Set <code className="text-zinc-300">WAKE_WORD_PYTHON</code> to use a
              different interpreter. Use Python 3.10–3.12 for TensorFlow. Restart the app after
              installing packages.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-md border border-border/50 bg-background/30 px-3 py-2 text-[11px] text-muted-foreground">
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-cyan-300/70" />
            Microphone recording works even before the Python environment is ready.
          </div>
        </div>
      </div>
    </div>
  );
}
