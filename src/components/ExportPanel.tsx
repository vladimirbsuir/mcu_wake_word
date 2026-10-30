import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  CheckCircle2,
  CircuitBoard,
  Download,
  FileCode2,
  FolderDown,
  Loader2,
  Package,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api, errorMessage } from "@/lib/api";
import { formatBytes, percent } from "@/lib/utils";
import type { ProjectSummary } from "@/types";

type ExportPanelProps = {
  summary: ProjectSummary;
  environmentReady: boolean;
  jobBusy: boolean;
  interactionDisabled: boolean;
};

const C_SNIPPET = `#include "wake_word_model.h"
#include "wake_word_config.h"
#include "wake_word_mel.h"

const uint8_t* model = model_data;
const size_t model_len = model_data_len;`;

export function ExportPanel({
  summary,
  environmentReady,
  jobBusy,
  interactionDisabled,
}: ExportPanelProps) {
  const [exporting, setExporting] = useState(false);
  const [exportedPath, setExportedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const metadata = summary.model.metadata;
  const tfliteSize =
    summary.model.tfliteSizeBytes || metadata?.model.tflite_size_bytes || 0;
  const disabled = exporting || jobBusy || interactionDisabled || !summary.model.exists;

  const exportPackage = async () => {
    setError(null);
    setExportedPath(null);
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Choose a folder for the ESP32 model package",
      });
      if (typeof selected !== "string") return;
      setExporting(true);
      const result = await api.exportForEsp32(summary.slug, selected);
      setExportedPath(result.destination);
    } catch (exportError) {
      setError(errorMessage(exportError));
    } finally {
      setExporting(false);
    }
  };

  return (
    <section id="deploy" className="panel relative scroll-mt-24 overflow-hidden p-5">
      <div className="pointer-events-none absolute -right-16 -top-16 h-44 w-44 rounded-full bg-[radial-gradient(circle,hsl(78_82%_61%/0.08),transparent_70%)]" />
      <div className="relative flex h-full flex-col">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-mono text-[10px] tracking-[0.2em] text-primary/70">05</span>
              <span className="label-caps">Deploy</span>
            </div>
            <h2 className="mt-2 text-lg font-semibold tracking-tight">Take it to silicon</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              A flash-ready C array and fully quantized TFLite model for TFLite Micro.
            </p>
          </div>
          <Badge variant="outline">ESP32-S3</Badge>
        </div>

        {!summary.model.exists ? (
          <div className="mt-5 flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-background/20 px-5 py-10 text-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-full border border-border bg-secondary/40">
              <CircuitBoard className="h-5 w-5 text-zinc-600" />
            </div>
            <p className="mt-3 text-xs font-medium text-zinc-400">No model package yet</p>
            <p className="mt-1.5 max-w-[230px] text-[11px] leading-relaxed text-muted-foreground">
              Finish a training run and the INT8 artifacts will appear here.
            </p>
          </div>
        ) : (
          <>
            <div className="mt-5 grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-border/70 bg-background/35 p-3">
                <div className="font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">
                  Flash model
                </div>
                <div className="mt-1.5 font-mono text-lg leading-none text-zinc-200">
                  {formatBytes(tfliteSize)}
                </div>
              </div>
              <div className="rounded-lg border border-border/70 bg-background/35 p-3">
                <div className="font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">
                  INT8 accuracy
                </div>
                <div className="mt-1.5 font-mono text-lg leading-none text-primary">
                  {percent(metadata?.metrics.tflite_test_accuracy ?? null)}
                </div>
              </div>
            </div>

            <div className="mt-2 rounded-lg border border-border/60 bg-background/25 p-3">
              <div className="flex items-center justify-between">
                <span className="label-caps">Package</span>
                <span className="flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-wider text-zinc-600">
                  <ShieldCheck className="h-3 w-3 text-cyan-300/60" />
                  {metadata?.input.dtype.toUpperCase() ?? "INT8"}
                </span>
              </div>
              <div className="mt-2.5 space-y-1.5">
                {[
                  ["wake_word_model.h", "C byte array"],
                  ["wake_word_config.h", "quantization + config"],
                  ["wake_word_mel.h", "window + mel filters"],
                  ["PREPROCESSING.md", "firmware guide"],
                  ["wake_word_int8.tflite", "quantized graph"],
                ].map(([file, hint]) => (
                  <div key={file} className="flex items-center gap-2">
                    <FileCode2 className="h-3 w-3 shrink-0 text-zinc-700" />
                    <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-zinc-400">
                      {file}
                    </span>
                    <span className="shrink-0 text-[9px] text-zinc-700">{hint}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-2 rounded-lg border border-border/50 bg-[#070a08] p-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">
                  Firmware usage
                </span>
                <Package className="h-3 w-3 text-zinc-700" />
              </div>
              <pre className="mt-2 overflow-x-auto font-mono text-[9.5px] leading-relaxed text-zinc-500">
                {C_SNIPPET}
              </pre>
            </div>

            <div className="mt-auto pt-4">
              <Button
                className="w-full"
                onClick={() => void exportPackage()}
                disabled={disabled}
                title={
                  !environmentReady
                    ? undefined
                    : jobBusy
                      ? "Wait for the current job to finish"
                      : undefined
                }
              >
                {exporting ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <FolderDown className="h-4 w-4" />
                )}
                {exporting ? "Copying package…" : "Export for ESP32"}
              </Button>
              <div className="mt-2.5 flex items-center justify-center gap-1.5 font-mono text-[8.5px] uppercase tracking-[0.1em] text-zinc-700">
                <CheckCircle2 className="h-2.5 w-2.5" />
                model_data[] · aligned 16 bytes · builtin int8 ops
              </div>

              {exportedPath ? (
                <div className="mt-3 rounded-md border border-primary/20 bg-primary/[0.045] px-3 py-2.5 text-[10.5px]">
                  <div className="flex items-center gap-1.5 text-primary">
                    <Download className="h-3 w-3" />
                    Package exported
                  </div>
                  <p className="mt-1 break-all font-mono text-[9.5px] leading-relaxed text-zinc-500">
                    {exportedPath}
                  </p>
                </div>
              ) : null}
              {error ? (
                <div className="mt-3 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[10.5px] leading-relaxed text-red-200">
                  {error}
                </div>
              ) : null}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
