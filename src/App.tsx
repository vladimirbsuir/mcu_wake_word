import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronRight,
  CircleDot,
  FolderPlus,
  Loader2,
  Lock,
  Pencil,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Square,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AiModelsDialog } from "@/components/AiModelsDialog";
import { DatasetPanel } from "@/components/DatasetPanel";
import { EnvironmentDialog } from "@/components/EnvironmentDialog";
import { ExportPanel } from "@/components/ExportPanel";
import { DeleteProjectDialog, ProjectNameDialog } from "@/components/ProjectDialogs";
import { PhraseLabPanel } from "@/components/PhraseLabPanel";
import { RecorderPanel } from "@/components/RecorderPanel";
import { Sidebar } from "@/components/Sidebar";
import { TrainingPanel } from "@/components/TrainingPanel";
import { JOB_TITLES, useJobRunner } from "@/hooks/useJobRunner";
import { ProjectVoiceProvider } from "@/hooks/useProjectVoice";
import { SpokenPhraseBar } from "@/components/SpokenPhraseBar";
import { LivePanel } from "@/components/LivePanel";
import { api, errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  categoryReadiness,
  guessLanguage,
  isUsable,
  voiceApi,
  type ModelsTab,
  type VoiceSettings,
  type VoiceStatus,
} from "@/lib/voice";
import type { EnvironmentStatus, ProjectListItem, ProjectSummary } from "@/types";

const PROJECT_STORAGE_KEY = "wakeforge.project";

type ProjectDialog =
  | { type: "create" }
  | { type: "rename"; project: ProjectListItem }
  | { type: "delete"; project: ProjectListItem }
  | null;

function storedProject(): string | null {
  try {
    return window.localStorage.getItem(PROJECT_STORAGE_KEY);
  } catch {
    // localStorage can be unavailable; the first project is used instead.
    return null;
  }
}

function storeProject(slug: string | null) {
  try {
    if (slug) window.localStorage.setItem(PROJECT_STORAGE_KEY, slug);
    else window.localStorage.removeItem(PROJECT_STORAGE_KEY);
  } catch {
    // Non-fatal.
  }
}

function WorkspaceSkeleton() {
  return (
    <div className="grid animate-pulse gap-4 xl:grid-cols-12">
      <div className="panel h-[570px] xl:col-span-8" />
      <div className="panel h-[570px] xl:col-span-4" />
      <div className="panel h-[520px] xl:col-span-8" />
      <div className="panel h-[520px] xl:col-span-4" />
    </div>
  );
}

function displayStage(stage: string | undefined, kind: string | null): string {
  if (stage) return stage.replaceAll("_", " ");
  if (kind === "augment") return "preparing dataset";
  if (kind === "train") return "training model";
  return "working";
}

export default function App() {
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(true);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [activeProject, setActiveProject] = useState<string | null>(storedProject);
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [projectLoading, setProjectLoading] = useState(true);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [dialog, setDialog] = useState<ProjectDialog>(null);
  const [recordingActive, setRecordingActive] = useState(false);
  const [environment, setEnvironment] = useState<EnvironmentStatus | null>(null);
  const [environmentLoading, setEnvironmentLoading] = useState(true);
  const [environmentError, setEnvironmentError] = useState<string | null>(null);
  const [environmentOpen, setEnvironmentOpen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus | null>(null);
  const [voiceStatusError, setVoiceStatusError] = useState<string | null>(null);
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings | null>(null);
  const [modelsDialog, setModelsDialog] = useState<{ open: boolean; category: ModelsTab }>({
    open: false,
    category: "llm",
  });
  const requestIdRef = useRef(0);

  const loadProjects = useCallback(async (): Promise<ProjectListItem[] | null> => {
    try {
      const list = await api.listProjects();
      setProjects(list);
      setProjectsError(null);
      return list;
    } catch (error) {
      setProjectsError(errorMessage(error));
      return null;
    } finally {
      setProjectsLoading(false);
    }
  }, []);

  // Pick the remembered project, or the most recent one, once the list is known.
  useEffect(() => {
    void (async () => {
      const list = await loadProjects();
      if (!list) return;
      setActiveProject((current) =>
        current && list.some((item) => item.slug === current) ? current : (list[0]?.slug ?? null),
      );
      if (!list.length) setDialog({ type: "create" });
    })();
  }, [loadProjects]);

  useEffect(() => {
    if (!projectsLoading) storeProject(activeProject);
  }, [activeProject, projectsLoading]);

  const loadProject = useCallback(
    async (showSpinner = true) => {
      const requestId = ++requestIdRef.current;
      if (!activeProject) {
        setSummary(null);
        setProjectError(null);
        setProjectLoading(false);
        return;
      }
      if (showSpinner) setProjectLoading(true);
      try {
        const next = await api.getProjectSummary(activeProject);
        if (requestId !== requestIdRef.current) return;
        setSummary(next);
        setProjectError(null);
      } catch (error) {
        if (requestId !== requestIdRef.current) return;
        setProjectError(errorMessage(error));
        setSummary(null);
      } finally {
        if (requestId === requestIdRef.current) setProjectLoading(false);
      }
    },
    [activeProject],
  );

  useEffect(() => {
    // Wait for the project list so a stale remembered id is never requested.
    if (!projectsLoading) void loadProject(true);
  }, [loadProject, projectsLoading]);

  // Takes, datasets and models change the counters shown in the project list.
  const refreshProject = useCallback(async () => {
    await Promise.all([loadProject(false), loadProjects()]);
  }, [loadProject, loadProjects]);

  const loadEnvironment = useCallback(async () => {
    setEnvironmentLoading(true);
    setEnvironmentError(null);
    try {
      const status = await api.getEnvironment();
      setEnvironment(status);
    } catch (error) {
      setEnvironmentError(errorMessage(error));
    } finally {
      setEnvironmentLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadEnvironment();
  }, [loadEnvironment]);

  const loadVoiceStatus = useCallback(async () => {
    try {
      setVoiceStatus(await voiceApi.status());
      setVoiceStatusError(null);
    } catch (error) {
      setVoiceStatusError(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void loadVoiceStatus();
    voiceApi
      .settings()
      .then(setVoiceSettings)
      .catch((error) => setVoiceStatusError(errorMessage(error)));
  }, [loadVoiceStatus]);

  const job = useJobRunner(refreshProject);

  // Downloads and package installs change what the catalog reports.
  useEffect(() => {
    if (job.finished && ["install", "packages", "corpus"].includes(job.finished.kind)) {
      void loadVoiceStatus();
    }
  }, [job.finished, loadVoiceStatus]);
  const environmentReady = Boolean(environment?.ready);
  const lockReason = recordingActive
    ? "Finish or discard the recording first."
    : // Global jobs (model downloads) do not touch project folders, so they do not lock switching.
      job.busy && job.kind && job.project
      ? `Wait for the ${JOB_TITLES[job.kind].toLowerCase()} job to finish or cancel it.`
      : null;
  const projectLanguage = summary
    ? summary.voice?.language || guessLanguage(summary.voice?.spokenText ?? summary.phrase)
    : null;
  const projectLanguageName =
    voiceStatus?.languages.find((item) => item.code === projectLanguage)?.name ?? null;
  const openModels = (category: ModelsTab) => setModelsDialog({ open: true, category });
  const activeItem = projects.find((item) => item.slug === activeProject) ?? null;

  const handleRecorderActivity = useCallback((active: boolean) => {
    setRecordingActive(active);
  }, []);

  const selectProject = useCallback(
    (slug: string | null) => {
      if (lockReason || slug === activeProject) return;
      // Logs, charts and results of the previous project's last job must not
      // show up in the next one.
      job.reset();
      setRecordingActive(false);
      setSummary(null);
      setProjectError(null);
      setProjectLoading(Boolean(slug));
      setActiveProject(slug);
    },
    [activeProject, job, lockReason],
  );

  const createProject = async (phrase: string) => {
    const created = await api.createProject(phrase);
    await loadProjects();
    setDialog(null);
    // Creating is allowed during a job; switching to the new project is not.
    selectProject(created.slug);
  };

  const renameProject = async (project: ProjectListItem, phrase: string) => {
    const previousSlug = project.slug;
    const renamed = await api.renameProject(previousSlug, phrase);
    await loadProjects();
    setDialog(null);
    if (previousSlug !== activeProject) return;
    if (renamed.slug === previousSlug) {
      await loadProject(false);
    } else {
      setSummary(null);
      setProjectLoading(true);
      setActiveProject(renamed.slug);
    }
  };

  const deleteProject = async (project: ProjectListItem) => {
    await api.deleteProject(project.slug);
    const list = (await loadProjects()) ?? [];
    setDialog(null);
    if (project.slug === activeProject) {
      job.reset();
      setSummary(null);
      setActiveProject(list[0]?.slug ?? null);
    }
  };

  const activeStage = displayStage(job.progress?.stage, job.kind);
  const jobPercent = Math.round((job.progress?.progress ?? 0) * 100);
  const projectBadge = useMemo(() => {
    if (!activeProject) return "NO PROJECT";
    if (!summary) return "LOADING";
    if (summary.model.exists) return "MODEL READY";
    if (summary.dataset.exists) return "DATASET READY";
    if (summary.recordings.length) return "CAPTURING";
    return "NEW PROJECT";
  }, [activeProject, summary]);

  const openEnvironment = () => setEnvironmentOpen(true);
  const openCreate = () => setDialog({ type: "create" });

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar
        projects={projects}
        projectsLoading={projectsLoading}
        activeProject={activeProject}
        lockReason={lockReason}
        onSelectProject={selectProject}
        onCreateProject={openCreate}
        onRenameProject={(project) => setDialog({ type: "rename", project })}
        onDeleteProject={(project) => setDialog({ type: "delete", project })}
        summary={summary}
        environment={environment}
        environmentLoading={environmentLoading}
        environmentError={environmentError}
        jobBusy={job.busy}
        jobKind={job.kind}
        onOpenEnvironment={openEnvironment}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 border-b border-border/70 bg-background/80 backdrop-blur-xl">
          <div className="flex h-14 items-center justify-between gap-4 px-6">
            <div className="flex min-w-0 items-center gap-2 font-mono text-[10px] uppercase tracking-[0.12em]">
              <span className="text-zinc-600">Workspace</span>
              <ChevronRight className="h-3 w-3 text-zinc-800" />
              <span className="truncate text-zinc-400">{activeProject ?? "—"}</span>
              <span className="ml-1 rounded border border-border/70 px-1.5 py-0.5 text-[8px] text-zinc-600">
                {projectBadge}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="hidden items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600 sm:flex">
                <ShieldCheck className="h-3 w-3 text-primary/50" />
                local first
              </div>
              <button
                type="button"
                onClick={() => openModels("llm")}
                className="flex h-7 items-center gap-1.5 rounded-md border border-border/60 bg-secondary/30 px-2.5 font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-500 transition-colors hover:border-border hover:text-zinc-300"
              >
                <Sparkles className="h-3 w-3" />
                AI models
              </button>
              <button
                type="button"
                onClick={openEnvironment}
                className="flex h-7 items-center gap-1.5 rounded-md border border-border/60 bg-secondary/30 px-2.5 font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-500 transition-colors hover:border-border hover:text-zinc-300"
              >
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    environment?.ready
                      ? "bg-primary"
                      : environmentLoading
                        ? "animate-pulse bg-zinc-500"
                        : "bg-amber-400",
                  )}
                />
                {environment?.ready ? "Python ready" : "Setup"}
              </button>
            </div>
          </div>
        </header>

        {job.busy ? (
          <div className="sticky top-14 z-20 border-b border-primary/10 bg-[#0b100c]/90 backdrop-blur-xl">
            <div className="flex min-h-10 items-center gap-3 px-6 py-2">
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
              <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.14em] text-primary/80">
                {job.kind ? JOB_TITLES[job.kind] : "Job"}
              </span>
              <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400">
                {job.progress?.message ?? `Starting ${activeStage}…`}
              </span>
              <span className="hidden font-mono text-[9px] uppercase tracking-[0.1em] text-zinc-600 md:inline">
                {activeStage}
              </span>
              <Progress
                value={jobPercent}
                className="hidden h-1 w-24 sm:block"
                indicatorClassName="bg-primary"
              />
              <span className="w-8 text-right font-mono text-[9px] text-zinc-500">{jobPercent}%</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-[9px] uppercase tracking-[0.1em] text-zinc-500 hover:text-red-300"
                onClick={() => void job.cancel()}
              >
                <Square className="h-2.5 w-2.5" />
                Cancel
              </Button>
            </div>
          </div>
        ) : null}

        <main className="flex-1">
          <div className="mx-auto w-full max-w-[1380px] px-6 pb-16">
            <section className="relative pb-7 pt-8">
              <div className="pointer-events-none absolute -left-24 top-0 h-56 w-[560px] bg-[radial-gradient(ellipse,hsl(180_70%_50%/0.03),transparent_65%)]" />
              <div className="relative flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
                <div className="max-w-2xl">
                  <div className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.2em] text-zinc-600">
                    <CircleDot className="h-3 w-3 text-primary/60" />
                    On-device voice · end-to-end
                  </div>
                  <h1 className="mt-3 text-[32px] font-semibold leading-[1.12] tracking-[-0.035em] text-zinc-100">
                    Custom wake words,
                    <br />
                    <span className="text-zinc-500">trained on your desk.</span>
                  </h1>
                  <p className="mt-3 max-w-xl text-[12.5px] leading-relaxed text-muted-foreground">
                    Record a phrase, expand it into a robust dataset, and compile a compact INT8
                    model. Everything runs locally; the final C header is ready for your firmware.
                  </p>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <Badge variant="outline">16 kHz · mono</Badge>
                    <Badge variant="outline">DS-CNN</Badge>
                    <Badge variant="outline">INT8 TFLite</Badge>
                    <Badge variant="outline">ESP32-S3</Badge>
                  </div>
                </div>

                <div className="w-full max-w-sm rounded-lg border border-border/70 bg-card/65 p-3.5">
                  <div className="flex items-center justify-between">
                    <span className="label-caps">Wake phrase / project</span>
                    {lockReason ? (
                      <span
                        className="flex items-center gap-1 font-mono text-[8px] uppercase tracking-[0.1em] text-zinc-600"
                        title={lockReason}
                      >
                        <Lock className="h-2.5 w-2.5" />
                        locked
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-2 flex items-center gap-1.5">
                    <div
                      className="min-w-0 flex-1 truncate text-[17px] font-semibold tracking-tight text-zinc-100"
                      title={summary?.phrase ?? activeItem?.phrase}
                    >
                      {summary?.phrase ?? activeItem?.phrase ?? "No project"}
                    </div>
                    {activeItem ? (
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => setDialog({ type: "rename", project: activeItem })}
                        disabled={Boolean(lockReason)}
                        title={lockReason ?? "Rename project"}
                        aria-label="Rename project"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                    ) : null}
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={openCreate}
                      title="New project"
                      aria-label="New project"
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-2">
                    <span className="truncate font-mono text-[9px] text-zinc-700">
                      projects/{activeProject ?? "—"}
                    </span>
                    <span className="shrink-0 font-mono text-[8.5px] uppercase tracking-[0.08em] text-zinc-700">
                      {summary
                        ? `${summary.recordings.length} takes`
                        : activeProject
                          ? "loading"
                          : `${projects.length} projects`}
                    </span>
                  </div>
                </div>
              </div>
            </section>

            {environment && !environment.ready && !environmentLoading ? (
              <div className="mb-4 flex flex-col gap-3 rounded-lg border border-amber-400/20 bg-amber-400/[0.045] px-4 py-3 sm:flex-row sm:items-center">
                <div className="flex min-w-0 flex-1 items-start gap-2.5">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-amber-100">The ML pipeline needs setup</p>
                    <p className="mt-0.5 text-[11px] leading-relaxed text-amber-100/50">
                      Recording works now. Dataset creation and training need TensorFlow and librosa
                      in a supported Python environment.
                    </p>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="shrink-0" onClick={openEnvironment}>
                  Open setup
                </Button>
              </div>
            ) : null}

            {environmentError && !environment ? (
              <div className="mb-4 flex items-center gap-3 rounded-lg border border-amber-400/20 bg-amber-400/[0.04] px-4 py-3 text-[11px] text-amber-100/70">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-300" />
                <span className="flex-1">{environmentError}</span>
                <Button variant="ghost" size="sm" onClick={() => void loadEnvironment()}>
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </Button>
              </div>
            ) : null}

            {projectsError ? (
              <div className="panel flex min-h-[320px] flex-col items-center justify-center p-8 text-center">
                <AlertTriangle className="h-5 w-5 text-amber-300" />
                <h2 className="mt-3 text-sm font-semibold">Cannot read the projects folder</h2>
                <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-muted-foreground">
                  {projectsError}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-4"
                  onClick={() => void loadProjects()}
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </Button>
              </div>
            ) : !projectsLoading && !activeProject ? (
              <div className="panel flex min-h-[320px] flex-col items-center justify-center p-8 text-center">
                <FolderPlus className="h-5 w-5 text-primary/70" />
                <h2 className="mt-3 text-sm font-semibold">Create a wake-word project</h2>
                <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-muted-foreground">
                  Each phrase gets its own takes, dataset, models and exports, so nothing from one
                  wake word leaks into another.
                </p>
                <Button size="sm" className="mt-4" onClick={openCreate}>
                  <Plus />
                  New project
                </Button>
              </div>
            ) : projectError && !summary ? (
              <div className="panel flex min-h-[320px] flex-col items-center justify-center p-8 text-center">
                <AlertTriangle className="h-5 w-5 text-amber-300" />
                <h2 className="mt-3 text-sm font-semibold">Cannot open this project</h2>
                <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-muted-foreground">
                  {projectError}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-4"
                  onClick={() => void refreshProject()}
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </Button>
              </div>
            ) : (projectLoading || projectsLoading) && !summary ? (
              <WorkspaceSkeleton />
            ) : summary ? (
              // Keyed by project: every panel starts with fresh local state.
              <ProjectVoiceProvider key={summary.slug} summary={summary} jobFinished={job.finished}>
              <div className="grid animate-fade-in gap-4 xl:grid-cols-12">
                <div className="min-w-0 xl:col-span-12">
                  <SpokenPhraseBar
                    summary={summary}
                    status={voiceStatus}
                    settings={voiceSettings}
                    jobBusy={job.busy}
                    jobKind={job.kind}
                    jobProject={job.project}
                    interactionDisabled={recordingActive}
                    onRunJob={(kind, starter) => void job.run(kind, summary.slug, starter)}
                    onOpenModels={openModels}
                  />
                </div>
                <div className="min-w-0 xl:col-span-12">
                  <RecorderPanel
                    summary={summary}
                    pipelineBusy={job.busy && Boolean(job.project)}
                    onRefresh={refreshProject}
                    onRecorderActivity={handleRecorderActivity}
                    llmReady={isUsable(categoryReadiness("llm", projectLanguage, voiceSettings, voiceStatus))}
                    wordsRunning={job.busy && job.kind === "words" && job.project === summary.slug}
                    onGenerateWords={(starter) => void job.run("words", summary.slug, starter)}
                    defaultGroupCounts={Object.fromEntries(
                      (voiceStatus?.groups ?? []).map((group) => [group.id, group.defaultCount]),
                    )}
                    onOpenModels={openModels}
                  />
                </div>
                <div className="min-w-0 xl:col-span-8">
                  <PhraseLabPanel
                    summary={summary}
                    status={voiceStatus}
                    settings={voiceSettings}
                    jobBusy={job.busy}
                    jobKind={job.kind}
                    jobProject={job.project}
                    jobProgress={job.progress}
                    jobFinished={job.finished}
                    jobError={job.error}
                    interactionDisabled={recordingActive}
                    onRunJob={(kind, starter) => void job.run(kind, summary.slug, starter)}
                    onCancelJob={() => void job.cancel()}
                    onOpenModels={openModels}
                    onRefresh={refreshProject}
                  />
                </div>
                <div className="min-w-0 xl:col-span-4">
                  <DatasetPanel
                    summary={summary}
                    corpora={voiceStatus?.corpora ?? []}
                    onOpenSpeechData={() => openModels("data")}
                    onDeleteMined={() =>
                      void api
                        .deleteMined(summary.slug)
                        .then(() => refreshProject())
                        .catch((error) => window.alert(errorMessage(error)))
                    }
                    environmentReady={environmentReady}
                    jobBusy={job.busy}
                    jobKind={job.kind}
                    progress={job.progress}
                    jobError={job.error}
                    interactionDisabled={recordingActive}
                    onStartAugment={(params) => void job.startAugmentation(params)}
                    onCancelJob={() => void job.cancel()}
                  />
                </div>
                <div className="min-w-0 xl:col-span-8">
                  <TrainingPanel
                    summary={summary}
                    corpora={voiceStatus?.corpora ?? []}
                    onOpenSpeechData={() => openModels("data")}
                    onStartMining={() =>
                      void job.run("mine", summary.slug, () => api.startMining(summary.slug))
                    }
                    onStartEvaluation={() =>
                      void job.run("evaluate", summary.slug, () => api.startEvaluation(summary.slug))
                    }
                    environmentReady={environmentReady}
                    jobBusy={job.busy}
                    jobKind={job.kind}
                    progress={job.progress}
                    liveHistory={job.liveHistory}
                    logs={job.logs}
                    finished={job.finished}
                    jobError={job.error}
                    interactionDisabled={recordingActive}
                    onStartTraining={(params) => void job.startTraining(params)}
                    onCancelJob={() => void job.cancel()}
                    onClearLogs={job.clearLogs}
                  />
                </div>
                <div className="min-w-0 xl:col-span-4">
                  <ExportPanel
                    summary={summary}
                    environmentReady={environmentReady}
                    jobBusy={job.busy}
                    interactionDisabled={recordingActive}
                  />
                </div>
                <div className="min-w-0 xl:col-span-12">
                  <LivePanel
                    summary={summary}
                    environmentReady={environmentReady}
                    jobBusy={job.busy}
                    jobKind={job.kind}
                    jobProject={job.project}
                    progress={job.progress}
                    finished={job.finished}
                    jobError={job.error}
                    interactionDisabled={recordingActive}
                    onStart={(params) => void job.run("live", summary.slug, () => api.startLive(params))}
                    onStop={() => void job.cancel()}
                    onRefresh={refreshProject}
                  />
                </div>
              </div>
              </ProjectVoiceProvider>
            ) : null}

            <footer className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t border-border/50 pt-4 font-mono text-[8.5px] uppercase tracking-[0.12em] text-zinc-800">
              <span>WakeForge Studio · local training workspace</span>
              <span>{summary?.paths.projectDir ?? "app data / projects"}</span>
            </footer>
          </div>
        </main>
      </div>

      {dialog?.type === "create" ? (
        <ProjectNameDialog
          mode="create"
          project={null}
          onClose={() => setDialog(null)}
          onSubmit={createProject}
        />
      ) : null}
      {dialog?.type === "rename" ? (
        <ProjectNameDialog
          mode="rename"
          project={dialog.project}
          onClose={() => setDialog(null)}
          onSubmit={(phrase) => renameProject(dialog.project, phrase)}
        />
      ) : null}
      {dialog?.type === "delete" ? (
        <DeleteProjectDialog
          project={dialog.project}
          onClose={() => setDialog(null)}
          onConfirm={() => deleteProject(dialog.project)}
        />
      ) : null}

      <AiModelsDialog
        open={modelsDialog.open}
        initialCategory={modelsDialog.category}
        language={projectLanguage}
        languageName={projectLanguageName}
        status={voiceStatus}
        statusError={voiceStatusError}
        settings={voiceSettings}
        jobBusy={job.busy}
        jobKind={job.kind}
        jobProgress={job.progress}
        jobFinished={job.finished}
        jobError={job.error}
        onRunJob={(kind, starter) => void job.run(kind, "", starter)}
        onCancelJob={() => void job.cancel()}
        onSettingsChange={setVoiceSettings}
        onReloadStatus={() => void loadVoiceStatus()}
        onClose={() => setModelsDialog((current) => ({ ...current, open: false }))}
      />

      <EnvironmentDialog
        open={environmentOpen}
        environment={environment}
        loading={environmentLoading}
        error={environmentError}
        onClose={() => setEnvironmentOpen(false)}
        onRefresh={() => void loadEnvironment()}
      />
    </div>
  );
}
