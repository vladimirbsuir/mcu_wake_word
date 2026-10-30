import { useEffect, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, FolderPen, FolderPlus, Loader2, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, errorMessage } from "@/lib/api";
import type { ProjectListItem, ProjectNamePreview } from "@/types";

const PREVIEW_DELAY_MS = 160;

type ModalShellProps = {
  title: string;
  icon: ReactNode;
  busy: boolean;
  onClose: () => void;
  children: ReactNode;
};

function ModalShell({ title, icon, busy, onClose, children }: ModalShellProps) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [busy, onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm animate-fade-in"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
      role="presentation"
    >
      <div
        className="panel w-full max-w-md bg-[#0c100e] p-5 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-md border border-border/60 bg-secondary/40 text-zinc-300">
              {icon}
            </span>
            <h2 className="text-sm font-semibold tracking-tight text-zinc-100">{title}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded p-1 text-zinc-600 transition-colors hover:text-zinc-300 disabled:opacity-40"
            aria-label="Close"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

type ProjectNameDialogProps = {
  mode: "create" | "rename";
  /** The project being renamed (rename mode only). */
  project: ProjectListItem | null;
  onClose: () => void;
  onSubmit: (phrase: string) => Promise<void>;
};

export function ProjectNameDialog({ mode, project, onClose, onSubmit }: ProjectNameDialogProps) {
  const [value, setValue] = useState(mode === "rename" ? (project?.phrase ?? "") : "");
  const [preview, setPreview] = useState<ProjectNamePreview | null>(null);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const requestRef = useRef(0);
  const exceptProject = mode === "rename" ? (project?.slug ?? null) : null;

  useEffect(() => {
    const requestId = ++requestRef.current;
    if (!value.trim()) {
      setPreview(null);
      setChecking(false);
      return;
    }
    setChecking(true);
    const timer = window.setTimeout(() => {
      api
        .previewProjectName(value, exceptProject)
        .then((next) => {
          if (requestId === requestRef.current) setPreview(next);
        })
        .catch((error) => {
          if (requestId === requestRef.current) {
            setPreview({ phrase: value.trim(), slug: "", error: errorMessage(error) });
          }
        })
        .finally(() => {
          if (requestId === requestRef.current) setChecking(false);
        });
    }, PREVIEW_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [value, exceptProject]);

  const unchanged = mode === "rename" && preview?.phrase === project?.phrase;
  const canSubmit =
    !saving && !checking && Boolean(preview) && !preview?.error && Boolean(preview?.slug) && !unchanged;
  const slugChanges = mode === "rename" && preview?.slug && preview.slug !== project?.slug;

  const submit = async () => {
    if (!canSubmit || !preview) return;
    setSaving(true);
    setSubmitError(null);
    try {
      await onSubmit(preview.phrase);
    } catch (error) {
      setSubmitError(errorMessage(error));
      setSaving(false);
    }
  };

  return (
    <ModalShell
      title={mode === "create" ? "New wake-word project" : "Rename project"}
      icon={mode === "create" ? <FolderPlus className="h-3.5 w-3.5" /> : <FolderPen className="h-3.5 w-3.5" />}
      busy={saving}
      onClose={onClose}
    >
      <div className="mt-4 space-y-1.5">
        <Label htmlFor="project-phrase">Wake phrase</Label>
        <Input
          id="project-phrase"
          value={value}
          autoFocus
          maxLength={64}
          spellCheck={false}
          placeholder="e.g. Среда, hey jarvis"
          onChange={(event) => {
            setValue(event.target.value);
            setSubmitError(null);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") void submit();
          }}
          disabled={saving}
          className="h-9 font-medium"
          aria-invalid={Boolean(preview?.error)}
        />
        <div className="flex min-h-4 items-center gap-1.5 font-mono text-[9.5px] text-zinc-600">
          {checking ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
          {preview?.slug ? <span className="truncate">projects/{preview.slug}</span> : null}
        </div>
      </div>

      {mode === "rename" ? (
        <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
          Takes, dataset and model stay with the project
          {slugChanges ? (
            <>
              ; the folder moves to <span className="font-mono text-zinc-300">{preview?.slug}</span>
            </>
          ) : null}
          . Retrain to update the phrase in exported headers. If the spoken phrase itself changes,
          create a new project instead.
        </p>
      ) : (
        <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
          Each project keeps its own takes, dataset, models and exports. Use the exact phrase you
          will say to the device.
        </p>
      )}

      {preview?.error || submitError ? (
        <div className="mt-3 flex gap-2 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[11px] leading-relaxed text-red-200">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{submitError ?? preview?.error}</span>
        </div>
      ) : null}

      <div className="mt-5 flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button size="sm" onClick={() => void submit()} disabled={!canSubmit}>
          {saving ? <Loader2 className="animate-spin" /> : null}
          {mode === "create" ? "Create project" : "Rename"}
        </Button>
      </div>
    </ModalShell>
  );
}

type DeleteProjectDialogProps = {
  project: ProjectListItem;
  onClose: () => void;
  onConfirm: () => Promise<void>;
};

export function DeleteProjectDialog({ project, onClose, onConfirm }: DeleteProjectDialogProps) {
  const [typed, setTyped] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasData = project.recordingCount > 0 || project.hasDataset || project.hasModel;
  // Typing the name is only required when there is something to lose.
  const confirmed = !hasData || typed.trim().toLowerCase() === project.phrase.toLowerCase();

  const remove = async () => {
    if (!confirmed || deleting) return;
    setDeleting(true);
    setError(null);
    try {
      await onConfirm();
    } catch (deleteError) {
      setError(errorMessage(deleteError));
      setDeleting(false);
    }
  };

  const contents = [
    `${project.recordingCount} recorded take${project.recordingCount === 1 ? "" : "s"}`,
    project.hasDataset ? "the augmented dataset" : null,
    project.hasModel ? "the trained model and C headers" : null,
  ].filter(Boolean);

  return (
    <ModalShell
      title="Delete project"
      icon={<Trash2 className="h-3.5 w-3.5 text-red-300" />}
      busy={deleting}
      onClose={onClose}
    >
      <p className="mt-4 text-xs leading-relaxed text-zinc-300">
        Permanently delete <span className="font-semibold text-zinc-100">“{project.phrase}”</span>?
      </p>
      <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">
        This removes {contents.join(", ")} from{" "}
        <span className="font-mono text-zinc-400">projects/{project.slug}</span>. It cannot be undone.
        Packages you already exported to other folders are not touched.
      </p>

      {hasData ? (
        <div className="mt-4 space-y-1.5">
          <Label htmlFor="delete-confirm">
            Type <span className="normal-case text-zinc-300">{project.phrase}</span> to confirm
          </Label>
          <Input
            id="delete-confirm"
            value={typed}
            autoFocus
            spellCheck={false}
            onChange={(event) => setTyped(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void remove();
            }}
            disabled={deleting}
            className="h-9"
          />
        </div>
      ) : null}

      {error ? (
        <div className="mt-3 flex gap-2 rounded-md border border-red-400/20 bg-red-400/[0.055] px-3 py-2 text-[11px] leading-relaxed text-red-200">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="mt-5 flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onClose} disabled={deleting}>
          Cancel
        </Button>
        <Button variant="destructive" size="sm" onClick={() => void remove()} disabled={!confirmed || deleting}>
          {deleting ? <Loader2 className="animate-spin" /> : <Trash2 />}
          Delete project
        </Button>
      </div>
    </ModalShell>
  );
}
