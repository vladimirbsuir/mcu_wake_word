import { useEffect, useRef, useState } from "react";
import { Check, Copy, Terminal, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { LogLine } from "@/hooks/useJobRunner";

type LogConsoleProps = {
  logs: LogLine[];
  busy: boolean;
  onClear: () => void;
  className?: string;
  maxHeight?: number;
};

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--:--:--";
  return date.toLocaleTimeString("en-GB", { hour12: false });
}

function lineClass(line: string, stream: string) {
  if (stream === "stderr") return "text-red-300/90";
  if (line.includes("WARNING") || line.includes("WARN")) return "text-amber-300/80";
  if (line.startsWith("[train]")) return "text-zinc-300";
  if (line.startsWith("[augment]")) return "text-lime-200/80";
  return "text-zinc-500";
}

export function LogConsole({ logs, busy, onClear, className, maxHeight = 220 }: LogConsoleProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node || !pinnedRef.current) return;
    node.scrollTop = node.scrollHeight;
  }, [logs]);

  const handleScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    pinnedRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 45;
  };

  const copy = async () => {
    if (!logs.length) return;
    const text = logs
      .map((log) => `${formatTime(log.timestamp)} ${log.stream === "stderr" ? "ERR" : "OUT"} ${log.line}`)
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard permissions can be denied inside some webviews.
    }
  };

  return (
    <div className={cn("overflow-hidden rounded-lg border border-border/80 bg-[#080b09]", className)}>
      <div className="flex h-9 items-center justify-between border-b border-border/70 bg-white/[0.018] px-3">
        <div className="flex items-center gap-2">
          <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            Pipeline console
          </span>
          <span
            className={cn(
              "ml-1 h-1.5 w-1.5 rounded-full",
              busy ? "animate-blink bg-primary" : "bg-zinc-700",
            )}
          />
        </div>
        <div className="flex items-center gap-1">
          <span className="mr-2 font-mono text-[10px] text-muted-foreground/50">
            {logs.length.toString().padStart(3, "0")} lines
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={copy}
            disabled={!logs.length}
            aria-label="Copy logs"
            title="Copy logs"
          >
            {copied ? <Check className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={onClear}
            disabled={!logs.length}
            aria-label="Clear logs"
            title="Clear console"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="overflow-y-auto px-3 py-2.5 font-mono text-[10.5px] leading-[1.65]"
        style={{ maxHeight }}
      >
        {!logs.length ? (
          <div className="flex h-[74px] flex-col items-center justify-center gap-2 text-center text-[11px] text-muted-foreground/40">
            <span className="text-muted-foreground/30">$ waiting for a pipeline</span>
            <span>Dataset and training output appears here in real time.</span>
          </div>
        ) : (
          logs.map((log) => (
            <div key={log.id} className="flex gap-2.5 whitespace-pre-wrap break-words">
              <span className="select-none text-zinc-700">{formatTime(log.timestamp)}</span>
              <span
                className={cn(
                  "select-none",
                  log.stream === "stderr" ? "text-red-400/60" : "text-zinc-700",
                )}
              >
                {log.stream === "stderr" ? "!" : "›"}
              </span>
              <span className={cn("min-w-0 flex-1", lineClass(log.line, log.stream))}>{log.line}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
