import {
  BrainCircuit,
  CircuitBoard,
  Layers,
  Lock,
  Mic,
  Pencil,
  Plus,
  Radio,
  Settings2,
  Trash2,
  Wand2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { EnvironmentStatus, ProjectListItem, ProjectSummary } from "@/types";
import type { PipelineKind } from "@/hooks/useJobRunner";

type SidebarProps = {
  projects: ProjectListItem[];
  projectsLoading: boolean;
  activeProject: string | null;
  /** Why switching, renaming and deleting are blocked right now, if they are. */
  lockReason: string | null;
  onSelectProject: (slug: string) => void;
  onCreateProject: () => void;
  onRenameProject: (project: ProjectListItem) => void;
  onDeleteProject: (project: ProjectListItem) => void;
  summary: ProjectSummary | null;
  environment: EnvironmentStatus | null;
  environmentLoading: boolean;
  environmentError: string | null;
  jobBusy: boolean;
  jobKind: PipelineKind;
  onOpenEnvironment: () => void;
};

type StepStatus = "done" | "current" | "upcoming";

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function BrandMark() {
  return (
    <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-primary/20 bg-primary/[0.09]">
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" aria-hidden="true">
        {[5, 9, 13, 17, 21].map((x, index) => {
          const heights = [5, 12, 18, 10, 6];
          const height = heights[index];
          return (
            <rect
              key={x}
              x={x - 1.1}
              y={12 - height / 2}
              width="2.2"
              height={height}
              rx="1.1"
              fill={index === 2 ? "#D6FF7C" : "#BAF24B"}
              opacity={index === 2 ? 1 : 0.72}
            />
          );
        })}
      </svg>
    </div>
  );
}

function ProjectRow({
  project,
  active,
  lockReason,
  onSelect,
  onRename,
  onDelete,
}: {
  project: ProjectListItem;
  active: boolean;
  lockReason: string | null;
  onSelect: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const locked = Boolean(lockReason);
  const switchBlocked = locked && !active;
  return (
    <div
      className={cn(
        "group relative flex items-center rounded-md transition-colors",
        active ? "bg-white/[0.05]" : "hover:bg-white/[0.025]",
      )}
    >
      {active ? (
        <span className="absolute left-0 top-1.5 h-[calc(100%-12px)] w-0.5 rounded bg-primary/70" />
      ) : null}
      <button
        type="button"
        onClick={onSelect}
        disabled={switchBlocked}
        title={switchBlocked ? (lockReason ?? undefined) : project.phrase}
        aria-current={active ? "true" : undefined}
        className="min-w-0 flex-1 rounded-md px-2.5 py-1.5 text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40"
      >
        <span
          className={cn(
            "block truncate text-[12px]",
            active ? "font-medium text-zinc-100" : "text-zinc-400",
          )}
        >
          {project.phrase}
        </span>
        <span className="mt-0.5 flex items-center gap-1.5 font-mono text-[8.5px] text-zinc-600">
          <span>{project.recordingCount} takes</span>
          <span className="h-0.5 w-0.5 rounded-full bg-zinc-700" />
          <span className={project.hasModel ? "text-primary/70" : undefined}>
            {project.hasModel ? "model" : project.hasDataset ? "dataset" : "no model"}
          </span>
        </span>
      </button>
      <div
        className={cn(
          "flex shrink-0 items-center gap-0.5 pr-1.5 transition-opacity",
          active
            ? "opacity-100"
            : "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
        )}
      >
        {[
          { label: "Rename", icon: Pencil, action: onRename, danger: false },
          { label: "Delete", icon: Trash2, action: onDelete, danger: true },
        ].map(({ label, icon: Icon, action, danger }) => (
          <button
            key={label}
            type="button"
            onClick={action}
            disabled={locked}
            title={locked ? (lockReason ?? undefined) : `${label} “${project.phrase}”`}
            aria-label={`${label} ${project.phrase}`}
            className={cn(
              "rounded p-1 text-zinc-600 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40",
              danger ? "hover:text-red-300" : "hover:text-zinc-200",
            )}
          >
            <Icon className="h-3 w-3" />
          </button>
        ))}
      </div>
    </div>
  );
}

export function Sidebar({
  projects,
  projectsLoading,
  activeProject,
  lockReason,
  onSelectProject,
  onCreateProject,
  onRenameProject,
  onDeleteProject,
  summary,
  environment,
  environmentLoading,
  environmentError,
  jobBusy,
  jobKind,
  onOpenEnvironment,
}: SidebarProps) {
  const recordingCount = summary?.recordings.length ?? 0;
  const datasetReady = Boolean(summary?.dataset.exists);
  const modelReady = Boolean(summary?.model.exists);

  const syntheticClips = summary?.synthetic
    ? summary.synthetic.counts.positive + summary.synthetic.counts.negative
    : 0;
  const phraseLabJob = jobKind === "transcribe" || jobKind === "words" || jobKind === "synthesize";

  let current: "capture" | "phrases" | "dataset" | "train" | "deploy" | "live" = "capture";
  if (jobKind === "live" && jobBusy) current = "live";
  else if (phraseLabJob && jobBusy) current = "phrases";
  else if (jobKind === "augment" && jobBusy) current = "dataset";
  else if (jobKind === "train" && jobBusy) current = "train";
  else if (modelReady) current = "deploy";
  else if (datasetReady) current = "train";
  else if (recordingCount > 0) current = "dataset";

  const steps: Array<{
    id: string;
    number: string;
    label: string;
    meta: string;
    icon: typeof Mic;
    status: StepStatus;
  }> = [
    {
      id: "capture",
      number: "01",
      label: "Capture",
      meta: recordingCount ? `${recordingCount} takes` : "record samples",
      icon: Mic,
      status: recordingCount >= 3 ? "done" : current === "capture" ? "current" : "upcoming",
    },
    {
      id: "phrases",
      number: "02",
      label: "Phrases",
      meta: syntheticClips ? `${syntheticClips} synthetic clips` : "optional · AI negatives",
      icon: Wand2,
      status: syntheticClips ? "done" : current === "phrases" ? "current" : "upcoming",
    },
    {
      id: "dataset",
      number: "03",
      label: "Dataset",
      meta: datasetReady ? `${summary?.dataset.total} clips` : "augment audio",
      icon: Layers,
      status: datasetReady ? "done" : current === "dataset" ? "current" : "upcoming",
    },
    {
      id: "train",
      number: "04",
      label: "Train",
      meta: modelReady ? "model ready" : "DS-CNN / INT8",
      icon: BrainCircuit,
      status: modelReady ? "done" : current === "train" ? "current" : "upcoming",
    },
    {
      id: "deploy",
      number: "05",
      label: "Deploy",
      meta: modelReady ? "export package" : "C header",
      icon: CircuitBoard,
      status: current === "deploy" && modelReady ? "current" : "upcoming",
    },
    {
      id: "live",
      number: "06",
      label: "Live test",
      meta: jobKind === "live" && jobBusy ? "listening" : "PC microphone",
      icon: Radio,
      status: current === "live" ? "current" : "upcoming",
    },
  ];

  const envReady = Boolean(environment?.ready);
  const envMessage = environmentLoading
    ? "Checking Python…"
    : environmentError
      ? "Environment check failed"
      : envReady
        ? `Python ${environment?.version} · ready`
        : environment
          ? "ML setup required"
          : "Environment unknown";

  return (
    <aside className="sticky top-0 flex h-screen w-[232px] shrink-0 flex-col border-r border-border/70 bg-[#080b09]/90 px-4 py-5 backdrop-blur">
      <div className="flex items-center gap-2.5 px-1">
        <BrandMark />
        <div>
          <div className="text-[13px] font-semibold leading-none tracking-tight text-zinc-100">
            WakeForge
          </div>
          <div className="mt-1 font-mono text-[8px] uppercase tracking-[0.22em] text-zinc-600">
            Embedded studio
          </div>
        </div>
      </div>

      <div className="mt-6 flex min-h-0 flex-col">
        <div className="flex items-center justify-between px-2">
          <span className="flex items-center gap-1.5">
            <span className="label-caps">Projects</span>
            {lockReason ? (
              <Lock className="h-2.5 w-2.5 text-zinc-600" aria-label={lockReason} />
            ) : null}
          </span>
          <button
            type="button"
            onClick={onCreateProject}
            className="flex h-5 w-5 items-center justify-center rounded text-zinc-500 transition-colors hover:bg-white/[0.05] hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            title="New project"
            aria-label="New project"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="mt-1.5 max-h-[212px] space-y-0.5 overflow-y-auto pr-0.5">
          {projects.map((project) => (
            <ProjectRow
              key={project.slug}
              project={project}
              active={project.slug === activeProject}
              lockReason={lockReason}
              onSelect={() => onSelectProject(project.slug)}
              onRename={() => onRenameProject(project)}
              onDelete={() => onDeleteProject(project)}
            />
          ))}
          {!projects.length ? (
            <button
              type="button"
              onClick={onCreateProject}
              className="w-full rounded-md border border-dashed border-border/70 px-2.5 py-2.5 text-left text-[11px] text-zinc-500 transition-colors hover:border-border hover:text-zinc-300"
            >
              {projectsLoading ? "Loading projects…" : "No projects yet · create one"}
            </button>
          ) : null}
        </div>
        {summary ? (
          <div className="mx-0.5 mt-2.5 flex items-center gap-3 rounded-md border border-border/50 bg-white/[0.015] px-2.5 py-2">
            {(
              [
                ["Takes", recordingCount.toString().padStart(2, "0"), false],
                ["Clips", String(summary.dataset.total), false],
                ["Model", modelReady ? "INT8" : "—", modelReady],
              ] as const
            ).map(([label, value, highlight], index) => (
              <div key={label} className="flex items-center gap-3">
                {index > 0 ? <div className="h-5 w-px bg-border/60" /> : null}
                <div>
                  <div className="font-mono text-[8.5px] uppercase tracking-wider text-zinc-700">
                    {label}
                  </div>
                  <div
                    className={cn(
                      "mt-0.5 font-mono text-[11px]",
                      highlight ? "uppercase text-primary/80" : "text-zinc-400",
                    )}
                  >
                    {value}
                  </div>
                </div>
              </div>
            ))}
            <span
              className={cn(
                "ml-auto h-1.5 w-1.5 rounded-full",
                jobBusy ? "animate-blink bg-cyan-300" : "bg-primary/60",
              )}
            />
          </div>
        ) : null}
      </div>

      <nav className="mt-6">
        <div className="label-caps px-2">Workflow</div>
        <div className="mt-2 space-y-0.5">
          {steps.map((step) => {
            const Icon = step.icon;
            const active = step.status === "current";
            return (
              <button
                key={step.id}
                type="button"
                onClick={() => scrollTo(step.id)}
                className={cn(
                  "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  active ? "bg-white/[0.04]" : "hover:bg-white/[0.025]",
                )}
              >
                <span
                  className={cn(
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-md border transition-colors",
                    active
                      ? "border-primary/25 bg-primary/[0.08] text-primary"
                      : step.status === "done"
                        ? "border-border/60 bg-secondary/40 text-zinc-400"
                        : "border-transparent text-zinc-600 group-hover:text-zinc-400",
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        "text-[12px]",
                        active ? "font-medium text-zinc-200" : "text-zinc-400",
                      )}
                    >
                      {step.label}
                    </span>
                    <span className="font-mono text-[8px] text-zinc-700">{step.number}</span>
                  </span>
                  <span className="mt-0.5 block truncate font-mono text-[9px] text-zinc-600">
                    {step.meta}
                  </span>
                </span>
                {step.status === "done" ? (
                  <span className="h-1 w-1 rounded-full bg-primary/60" />
                ) : active ? (
                  <span className="h-1 w-1 animate-pulse rounded-full bg-primary" />
                ) : null}
              </button>
            );
          })}
        </div>
      </nav>

      <div className="mt-5 rounded-lg border border-border/50 bg-background/20 p-3">
        <div className="flex items-center gap-1.5">
          <CircuitBoard className="h-3 w-3 text-zinc-600" />
          <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-zinc-600">
            Target profile
          </span>
        </div>
        <div className="mt-2 flex items-center justify-between font-mono text-[9.5px]">
          <span className="text-zinc-500">Chip</span>
          <span className="text-zinc-300">ESP32-S3</span>
        </div>
        <div className="mt-1 flex items-center justify-between font-mono text-[9.5px]">
          <span className="text-zinc-500">Runtime</span>
          <span className="text-zinc-300">TFLite Micro</span>
        </div>
        <div className="mt-1 flex items-center justify-between font-mono text-[9.5px]">
          <span className="text-zinc-500">Precision</span>
          <span className="text-zinc-300">full INT8</span>
        </div>
      </div>

      <div className="mt-auto space-y-2 pt-5">
        <button
          type="button"
          onClick={onOpenEnvironment}
          className="flex w-full items-center gap-2.5 rounded-md border border-border/60 bg-background/25 px-2.5 py-2 text-left transition-colors hover:border-border hover:bg-secondary/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <span
            className={cn(
              "h-1.5 w-1.5 shrink-0 rounded-full",
              envReady
                ? "bg-primary shadow-[0_0_7px_hsl(78_82%_61%/0.6)]"
                : environmentLoading
                  ? "animate-pulse bg-zinc-500"
                  : "bg-amber-400",
            )}
          />
          <span className="min-w-0 flex-1">
            <span className="block text-[10.5px] text-zinc-400">ML environment</span>
            <span className="mt-0.5 block truncate font-mono text-[8.5px] text-zinc-600">
              {envMessage}
            </span>
          </span>
          <Settings2 className="h-3 w-3 shrink-0 text-zinc-700" />
        </button>
        <div className="flex items-center justify-between px-1 font-mono text-[8px] uppercase tracking-[0.12em] text-zinc-800">
          <span>WakeForge v0.1.0</span>
          <span>local only</span>
        </div>
      </div>
    </aside>
  );
}
