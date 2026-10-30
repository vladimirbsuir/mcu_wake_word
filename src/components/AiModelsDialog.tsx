import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  AudioLines,
  Check,
  CircleCheck,
  Cloud,
  Download,
  ExternalLink,
  HardDrive,
  KeyRound,
  Loader2,
  Package,
  Pencil,
  Plus,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { JOB_TITLES, type PipelineKind } from "@/hooks/useJobRunner";
import { errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  categoryReadiness,
  LOCAL_PROVIDER,
  modelReadiness,
  newProfileId,
  profileServes,
  supportsLanguage,
  VOICE_CATEGORIES,
  voiceApi,
  type CatalogModel,
  type ModelsTab,
  type SpeechCorpus,
  type VoiceCategory,
  type VoiceProfile,
  type VoiceSettings,
  type VoiceStatus,
} from "@/lib/voice";
import type { JobFinishedEvent, JobProgress, JobStarted } from "@/types";

const VOICE_JOBS: PipelineKind[] = ["install", "packages", "check", "corpus"];

// Presets fill in only well-known base URLs; model names change too often to hard-code.
const ENDPOINT_PRESETS: Array<{ name: string; baseUrl: string }> = [
  { name: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  { name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1" },
  { name: "Groq", baseUrl: "https://api.groq.com/openai/v1" },
  { name: "Ollama (local)", baseUrl: "http://localhost:11434/v1" },
  { name: "LM Studio (local)", baseUrl: "http://localhost:1234/v1" },
];

type AiModelsDialogProps = {
  open: boolean;
  initialCategory: ModelsTab;
  /** Language of the current project's phrase, used for compatibility hints. */
  language: string | null;
  languageName: string | null;
  status: VoiceStatus | null;
  statusError: string | null;
  settings: VoiceSettings | null;
  jobBusy: boolean;
  jobKind: PipelineKind;
  jobProgress: JobProgress | null;
  jobFinished: JobFinishedEvent | null;
  jobError: string | null;
  onRunJob: (kind: Exclude<PipelineKind, null>, starter: () => Promise<JobStarted>) => void;
  onCancelJob: () => void;
  onSettingsChange: (settings: VoiceSettings) => void;
  onReloadStatus: () => void;
  onClose: () => void;
};

function formatSize(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function languageSummary(languages: string[] | null): string {
  if (!languages) return "multilingual";
  if (languages.length > 6) return `${languages.length} languages`;
  return languages.join(" · ");
}

type ProfileEditorProps = {
  profile: VoiceProfile;
  isNew: boolean;
  onCancel: () => void;
  onSave: (profile: VoiceProfile, key: string | null) => Promise<void>;
  onDelete: () => Promise<void>;
};

function ProfileEditor({ profile, isNew, onCancel, onSave, onDelete }: ProfileEditorProps) {
  const [draft, setDraft] = useState(profile);
  const [key, setKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = (patch: Partial<VoiceProfile>) => setDraft((current) => ({ ...current, ...patch }));
  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await onSave(draft, key.trim() ? key.trim() : null);
    } catch (saveError) {
      setError(errorMessage(saveError));
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3 rounded-lg border border-border/70 bg-background/40 p-3.5">
      <div className="flex flex-wrap gap-1.5">
        {ENDPOINT_PRESETS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            onClick={() => update({ name: draft.name || preset.name, baseUrl: preset.baseUrl })}
            className="rounded border border-border/60 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-zinc-500 hover:border-border hover:text-zinc-200"
          >
            {preset.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        <div className="space-y-1">
          <Label htmlFor="profile-name">Name</Label>
          <Input id="profile-name" value={draft.name} onChange={(event) => update({ name: event.target.value })} className="h-8" />
        </div>
        <div className="space-y-1">
          <Label htmlFor="profile-url">Base URL</Label>
          <Input
            id="profile-url"
            value={draft.baseUrl}
            placeholder="https://api.example.com/v1"
            onChange={(event) => update({ baseUrl: event.target.value })}
            className="h-8 font-mono text-xs"
          />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2.5">
        {VOICE_CATEGORIES.map((category) => (
          <div key={category.id} className="space-y-1">
            <Label htmlFor={`profile-model-${category.id}`}>{category.short} model</Label>
            <Input
              id={`profile-model-${category.id}`}
              value={draft.models[category.id] ?? ""}
              placeholder="not used"
              onChange={(event) => update({ models: { ...draft.models, [category.id]: event.target.value } })}
              className="h-8 font-mono text-xs"
            />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        <div className="space-y-1">
          <Label htmlFor="profile-voices">TTS voices (comma-separated)</Label>
          <Input
            id="profile-voices"
            value={draft.voices.join(", ")}
            placeholder="provider defaults"
            onChange={(event) =>
              update({ voices: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })
            }
            className="h-8 font-mono text-xs"
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="profile-languages">Languages (codes, optional)</Label>
          <Input
            id="profile-languages"
            value={draft.languages.join(", ")}
            placeholder="any, e.g. ru, en"
            onChange={(event) =>
              update({
                languages: event.target.value
                  .split(",")
                  .map((value) => value.trim().toLowerCase())
                  .filter(Boolean),
              })
            }
            className="h-8 font-mono text-xs"
          />
        </div>
      </div>
      <div className="space-y-1">
        <Label htmlFor="profile-key">API key</Label>
        <Input
          id="profile-key"
          type="password"
          autoComplete="off"
          value={key}
          placeholder={profile.hasKey ? "Saved · type to replace" : "Not needed for local servers"}
          onChange={(event) => setKey(event.target.value)}
          className="h-8 font-mono text-xs"
        />
        <p className="flex items-center gap-1 text-[10px] text-zinc-600">
          <KeyRound className="h-2.5 w-2.5" />
          Stored in the OS credential store, never in project files. It is sent only to this URL.
        </p>
      </div>
      {error ? <p className="text-[11px] text-red-300">{error}</p> : null}
      <div className="flex items-center justify-between">
        {!isNew ? (
          <Button variant="ghost" size="sm" className="text-zinc-500 hover:text-red-300" onClick={() => void onDelete()}>
            <Trash2 className="h-3 w-3" /> Delete endpoint
          </Button>
        ) : (
          <span />
        )}
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={saving}>
            Cancel
          </Button>
          <Button size="sm" onClick={() => void save()} disabled={saving || !draft.name.trim() || !draft.baseUrl.trim()}>
            {saving ? <Loader2 className="animate-spin" /> : <Check />}
            Save endpoint
          </Button>
        </div>
      </div>
    </div>
  );
}

export function AiModelsDialog({
  open,
  initialCategory,
  language,
  languageName,
  status,
  statusError,
  settings,
  jobBusy,
  jobKind,
  jobProgress,
  jobFinished,
  jobError,
  onRunJob,
  onCancelJob,
  onSettingsChange,
  onReloadStatus,
  onClose,
}: AiModelsDialogProps) {
  const [tab, setTab] = useState<ModelsTab>(initialCategory);
  // Provider hooks below need a category even while the speech data tab is open.
  const category: VoiceCategory = tab === "data" ? "stt" : tab;
  const [editing, setEditing] = useState<{ profile: VoiceProfile; isNew: boolean } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const voiceJobRunning = jobBusy && VOICE_JOBS.includes(jobKind);

  useEffect(() => {
    if (open) {
      setTab(initialCategory);
      setActionError(null);
      setEditing(null);
    }
  }, [open, initialCategory]);

  useEffect(() => {
    if (!open) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !editing) onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, editing, onClose]);

  const models = useMemo(
    () => (status?.models ?? []).filter((model) => model.category === category),
    [status, category],
  );
  const profiles = settings?.profiles ?? [];
  const active = settings?.active?.[category] ?? null;

  if (!open) return null;

  const persist = async (next: VoiceSettings) => {
    setActionError(null);
    try {
      onSettingsChange(await voiceApi.saveSettings(next));
    } catch (error) {
      setActionError(errorMessage(error));
    }
  };

  const choose = (provider: string, model?: string) => {
    if (!settings) return;
    void persist({ ...settings, active: { ...settings.active, [category]: { provider, model } } });
  };

  const removeCorpus = async (corpus: SpeechCorpus) => {
    if (!window.confirm(`Delete the prepared audio of ${corpus.title}?`)) return;
    setActionError(null);
    try {
      await voiceApi.removeCorpus(corpus.id);
      onReloadStatus();
    } catch (error) {
      setActionError(errorMessage(error));
    }
  };

  const removeModel = async (model: CatalogModel) => {
    if (!window.confirm(`Delete the downloaded files of ${model.title} (${formatSize(model.sizeMb)})?`)) return;
    setActionError(null);
    try {
      await voiceApi.removeModel(model.id);
      if (active?.provider === LOCAL_PROVIDER && active.model === model.id && settings) {
        await persist({ ...settings, active: { ...settings.active, [category]: null } });
      }
      onReloadStatus();
    } catch (error) {
      setActionError(errorMessage(error));
    }
  };

  const saveProfile = async (profile: VoiceProfile, key: string | null) => {
    if (!settings) return;
    const exists = settings.profiles.some((item) => item.id === profile.id);
    const nextProfiles = exists
      ? settings.profiles.map((item) => (item.id === profile.id ? profile : item))
      : [...settings.profiles, profile];
    let saved = await voiceApi.saveSettings({ ...settings, profiles: nextProfiles });
    if (key) saved = await voiceApi.setApiKey(profile.id, key);
    onSettingsChange(saved);
    setEditing(null);
  };

  const deleteProfile = async (profile: VoiceProfile) => {
    if (!settings || !window.confirm(`Delete endpoint “${profile.name}” and its stored API key?`)) return;
    await voiceApi.setApiKey(profile.id, null);
    onSettingsChange(
      await voiceApi.saveSettings({ ...settings, profiles: settings.profiles.filter((item) => item.id !== profile.id) }),
    );
    setEditing(null);
  };

  const readiness = categoryReadiness(category, language, settings, status);
  const info = VOICE_CATEGORIES.find((item) => item.id === category)!;
  const lastResult =
    jobFinished && VOICE_JOBS.includes(jobFinished.kind as PipelineKind) ? jobFinished : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm animate-fade-in"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !editing) onClose();
      }}
      role="presentation"
    >
      <div
        className="panel flex max-h-[90vh] w-full max-w-3xl flex-col bg-[#0c100e] shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="AI models"
      >
        <div className="flex items-start justify-between gap-4 border-b border-border/60 p-5">
          <div>
            <h2 className="text-base font-semibold tracking-tight text-zinc-100">AI models</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Optional helpers that write and voice negative phrases, plus downloadable real speech.
              One provider is active per category; local models run on this PC, endpoints call an API.
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded p-1 text-zinc-600 hover:text-zinc-300" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex gap-1 border-b border-border/60 px-5 pt-3">
          {VOICE_CATEGORIES.map((item) => {
            const state = categoryReadiness(item.id, language, settings, status).state;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setTab(item.id);
                  setEditing(null);
                }}
                className={cn(
                  "-mb-px flex items-center gap-2 rounded-t-md border border-b-0 px-3 py-2 text-xs transition-colors",
                  tab === item.id
                    ? "border-border/70 bg-card text-zinc-100"
                    : "border-transparent text-zinc-500 hover:text-zinc-300",
                )}
              >
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    state === "ready" || state === "unverified"
                      ? "bg-primary"
                      : state === "unselected"
                        ? "bg-zinc-600"
                        : "bg-amber-400",
                  )}
                />
                {item.title}
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => {
              setTab("data");
              setEditing(null);
            }}
            className={cn(
              "-mb-px flex items-center gap-2 rounded-t-md border border-b-0 px-3 py-2 text-xs transition-colors",
              tab === "data"
                ? "border-border/70 bg-card text-zinc-100"
                : "border-transparent text-zinc-500 hover:text-zinc-300",
            )}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                status?.corpora?.some((corpus) => corpus.installed) ? "bg-primary" : "bg-amber-400",
              )}
            />
            Speech data
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
            <span>
              {tab === "data"
                ? "Hours of real speech from many people. Training sets become negatives; benchmark sets are only used by the streaming test."
                : info.purpose}
            </span>
            {language ? (
              <Badge variant="outline">
                project language: {languageName ?? language}
              </Badge>
            ) : null}
          </div>

          {statusError ? (
            <div className="flex gap-2 rounded-md border border-amber-400/20 bg-amber-400/[0.05] px-3 py-2 text-[11px] text-amber-100/80">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300" />
              <span>{statusError}</span>
            </div>
          ) : null}

          {voiceJobRunning ? (
            <div className="space-y-2 rounded-md border border-cyan-300/15 bg-cyan-300/[0.035] px-3 py-2.5">
              <div className="flex items-center justify-between gap-2 text-[11px] text-cyan-100/80">
                <span className="flex min-w-0 items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                  <span className="truncate">{jobProgress?.message ?? `${JOB_TITLES[jobKind!]}…`}</span>
                </span>
                <Button variant="ghost" size="sm" className="h-6 px-2 text-[10px]" onClick={onCancelJob}>
                  Cancel
                </Button>
              </div>
              <Progress value={(jobProgress?.progress ?? 0) * 100} className="h-1" indicatorClassName="bg-cyan-300" />
            </div>
          ) : lastResult ? (
            <div
              className={cn(
                "flex gap-2 rounded-md border px-3 py-2 text-[11px]",
                lastResult.success
                  ? "border-primary/20 bg-primary/[0.05] text-zinc-300"
                  : "border-red-400/20 bg-red-400/[0.05] text-red-200",
              )}
            >
              {lastResult.success ? <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" /> : <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
              <span className="min-w-0 break-words">
                {JOB_TITLES[lastResult.kind as Exclude<PipelineKind, null>]}:{" "}
                {lastResult.success
                  ? lastResult.kind === "check"
                    ? `works · ${JSON.stringify(lastResult.result ?? {})}`
                    : "done"
                  : (jobError ?? lastResult.message)}
              </span>
            </div>
          ) : null}

          {actionError ? <p className="text-[11px] text-red-300">{actionError}</p> : null}

          {tab === "data" ? (
            <CorporaSection
              corpora={status?.corpora ?? null}
              language={language}
              jobBusy={jobBusy}
              onInstall={(corpus) => onRunJob("corpus", () => voiceApi.installCorpus(corpus.id))}
              onRemove={(corpus) => void removeCorpus(corpus)}
            />
          ) : (
          <>
          <section>
            <div className="mb-2 flex items-center gap-1.5 label-caps">
              <HardDrive className="h-3 w-3" /> Local models
            </div>
            {!status ? (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Reading the model catalog…
              </div>
            ) : (
              <div className="space-y-2">
                {models.map((model) => {
                  const state = modelReadiness(model, status);
                  const supported = supportsLanguage(model.languages, language);
                  const selected = active?.provider === LOCAL_PROVIDER && active.model === model.id;
                  const recommended =
                    supported &&
                    (model.recommendedFor.includes("*") || (language ? model.recommendedFor.includes(language) : false));
                  const runtime = model.dependsOn
                    .map((id) => status.models.find((item) => item.id === id))
                    .filter((item): item is CatalogModel => Boolean(item && !item.installed));
                  const downloadMb = model.sizeMb + runtime.reduce((sum, item) => sum + item.sizeMb, 0);
                  const missingPackages = model.requirements.filter((item) => !item.installed);
                  return (
                    <div
                      key={model.id}
                      className={cn(
                        "rounded-lg border p-3 transition-colors",
                        selected ? "border-primary/40 bg-primary/[0.04]" : "border-border/60 bg-background/25",
                        !supported && "opacity-60",
                      )}
                    >
                      <div className="flex items-start gap-3">
                        <button
                          type="button"
                          onClick={() => choose(LOCAL_PROVIDER, model.id)}
                          disabled={state !== "ready" || selected}
                          title={state !== "ready" ? "Download the model first" : "Use this model"}
                          className={cn(
                            "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                            selected ? "border-primary bg-primary" : "border-zinc-600",
                            state !== "ready" && "cursor-not-allowed opacity-40",
                          )}
                          aria-label={`Use ${model.title}`}
                        >
                          {selected ? <span className="h-1.5 w-1.5 rounded-full bg-black" /> : null}
                        </button>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="text-[13px] font-medium text-zinc-100">{model.title}</span>
                            {recommended ? <Badge variant="cyan">recommended</Badge> : null}
                            {!supported ? <Badge variant="outline">no {language}</Badge> : null}
                          </div>
                          <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">{model.description}</p>
                          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[9.5px] text-zinc-600">
                            <span>{languageSummary(model.languages)}</span>
                            <span>{formatSize(model.sizeMb)}</span>
                            {model.ramGb ? <span>~{model.ramGb} GB RAM</span> : null}
                            {model.speakers > 1 ? <span>{model.speakers} voices</span> : null}
                            <span>{model.license}</span>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          {state === "download" ? (
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={jobBusy}
                              onClick={() => onRunJob("install", () => voiceApi.installModel(model.id))}
                              title={runtime.length ? `Also downloads ${runtime.map((item) => item.title).join(", ")}` : undefined}
                            >
                              <Download className="h-3 w-3" /> {formatSize(downloadMb)}
                            </Button>
                          ) : null}
                          {state === "packages" ? (
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={jobBusy}
                              onClick={() => onRunJob("packages", () => voiceApi.installPackages(model.id))}
                              title={missingPackages.map((item) => item.package).join(", ")}
                            >
                              <Package className="h-3 w-3" /> Install {missingPackages.map((item) => item.package.split(/[=<>]/)[0]).join(", ")}
                            </Button>
                          ) : null}
                          {model.installed ? (
                            <button
                              type="button"
                              onClick={() => void removeModel(model)}
                              disabled={jobBusy}
                              className="rounded p-1.5 text-zinc-600 hover:text-red-300 disabled:opacity-40"
                              title="Delete downloaded files"
                              aria-label={`Delete ${model.title}`}
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          <section>
            <div className="mb-2 flex items-center justify-between">
              <span className="flex items-center gap-1.5 label-caps">
                <Cloud className="h-3 w-3" /> API endpoints
              </span>
              {!editing ? (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-[11px]"
                  onClick={() =>
                    setEditing({
                      isNew: true,
                      profile: {
                        id: newProfileId(),
                        type: "openai",
                        name: "",
                        baseUrl: "",
                        models: { llm: "", tts: "", stt: "" },
                        voices: [],
                        languages: [],
                        hasKey: false,
                      },
                    })
                  }
                >
                  <Plus className="h-3 w-3" /> Add endpoint
                </Button>
              ) : null}
            </div>
            {editing ? (
              <ProfileEditor
                profile={editing.profile}
                isNew={editing.isNew}
                onCancel={() => setEditing(null)}
                onSave={saveProfile}
                onDelete={() => deleteProfile(editing.profile)}
              />
            ) : (
              <div className="space-y-2">
                {profiles.map((profile) => {
                  const serves = profileServes(profile, category);
                  const selected = active?.provider === profile.id;
                  const supported = supportsLanguage(profile.languages, language);
                  return (
                    <div
                      key={profile.id}
                      className={cn(
                        "flex items-center gap-3 rounded-lg border p-3",
                        selected ? "border-primary/40 bg-primary/[0.04]" : "border-border/60 bg-background/25",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => choose(profile.id)}
                        disabled={!serves || selected}
                        title={serves ? "Use this endpoint" : `Set a ${info.short} model for this endpoint first`}
                        className={cn(
                          "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                          selected ? "border-primary bg-primary" : "border-zinc-600",
                          !serves && "cursor-not-allowed opacity-40",
                        )}
                        aria-label={`Use ${profile.name}`}
                      >
                        {selected ? <span className="h-1.5 w-1.5 rounded-full bg-black" /> : null}
                      </button>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-1.5 text-[13px] text-zinc-100">
                          {profile.name}
                          {profile.hasKey ? <KeyRound className="h-3 w-3 text-zinc-500" /> : null}
                          {!supported ? <Badge variant="outline">no {language}</Badge> : null}
                        </div>
                        <div className="mt-0.5 truncate font-mono text-[9.5px] text-zinc-600">
                          {profile.baseUrl} · {serves ? profile.models[category] : `no ${info.short} model`}
                          {profile.languages.length ? ` · ${profile.languages.join(", ")}` : " · languages not declared"}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setEditing({ profile, isNew: false })}
                        className="rounded p-1.5 text-zinc-600 hover:text-zinc-200"
                        aria-label={`Edit ${profile.name}`}
                      >
                        <Pencil className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })}
                {!profiles.length ? (
                  <p className="text-[11px] text-zinc-600">
                    No endpoints yet. Add OpenAI, OpenRouter, Groq, or a local Ollama / LM Studio server.
                  </p>
                ) : null}
              </div>
            )}
          </section>
          </>
          )}
        </div>

        {tab !== "data" ? (
        <div className="flex items-center justify-between gap-3 border-t border-border/60 px-5 py-3">
          <span className="min-w-0 truncate text-[11px] text-zinc-500">
            {readiness.provider ? `${info.short}: ${readiness.provider} · ${readiness.label}` : `${info.short}: nothing selected`}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={jobBusy || !(readiness.state === "ready" || readiness.state === "unverified")}
            onClick={() => onRunJob("check", () => voiceApi.check(category, language))}
          >
            <Zap className="h-3 w-3" /> Test {info.short}
          </Button>
        </div>
        ) : null}
      </div>
    </div>
  );
}

type CorporaSectionProps = {
  corpora: SpeechCorpus[] | null;
  language: string | null;
  jobBusy: boolean;
  onInstall: (corpus: SpeechCorpus) => void;
  onRemove: (corpus: SpeechCorpus) => void;
};

const CORPUS_ROLES: Array<{ role: SpeechCorpus["role"]; title: string; hint: string }> = [
  {
    role: "negatives",
    title: "Training negatives",
    hint: "Cut into windows and added to every dataset split as speech that must not trigger.",
  },
  {
    role: "eval",
    title: "Benchmark",
    hint: "Never trained on: the streaming test counts false activations per hour on it.",
  },
];

function CorporaSection({ corpora, language, jobBusy, onInstall, onRemove }: CorporaSectionProps) {
  if (!corpora) {
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Reading the speech data catalog…
      </div>
    );
  }
  return (
    <>
      {CORPUS_ROLES.map(({ role, title, hint }) => (
        <section key={role}>
          <div className="mb-1 flex items-center gap-1.5 label-caps">
            <AudioLines className="h-3 w-3" /> {title}
          </div>
          <p className="mb-2 text-[11px] text-zinc-600">{hint}</p>
          <div className="space-y-2">
            {corpora
              .filter((corpus) => corpus.role === role)
              .map((corpus) => {
                const otherLanguage = Boolean(language) && corpus.language !== language;
                return (
                  <div
                    key={corpus.id}
                    className={cn(
                      "flex items-start gap-3 rounded-lg border p-3",
                      corpus.installed ? "border-primary/30 bg-primary/[0.03]" : "border-border/60 bg-background/25",
                      otherLanguage && "opacity-60",
                    )}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[13px] font-medium text-zinc-100">{corpus.title}</span>
                        <Badge variant="outline">{corpus.language}</Badge>
                        {corpus.installed ? (
                          <Badge variant="default">
                            <Check className="h-2.5 w-2.5" /> {corpus.installedHours ?? corpus.hours} h
                          </Badge>
                        ) : null}
                      </div>
                      <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">{corpus.description}</p>
                      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[9.5px] text-zinc-600">
                        <span>~{corpus.hours} h</span>
                        <span>{formatSize(corpus.sizeMb)} download</span>
                        {corpus.utterances ? <span>{corpus.utterances.toLocaleString()} utterances</span> : null}
                        <span>{corpus.license}</span>
                        <a
                          href={corpus.homepage}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-0.5 hover:text-zinc-300"
                        >
                          source <ExternalLink className="h-2.5 w-2.5" />
                        </a>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      {!corpus.installed ? (
                        <Button variant="outline" size="sm" disabled={jobBusy} onClick={() => onInstall(corpus)}>
                          <Download className="h-3 w-3" /> {formatSize(corpus.sizeMb)}
                        </Button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onRemove(corpus)}
                          disabled={jobBusy}
                          className="rounded p-1.5 text-zinc-600 hover:text-red-300 disabled:opacity-40"
                          title="Delete prepared audio"
                          aria-label={`Delete ${corpus.title}`}
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        </section>
      ))}
    </>
  );
}
