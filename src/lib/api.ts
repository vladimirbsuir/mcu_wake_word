import { invoke } from "@tauri-apps/api/core";
import type {
  CaptureStart,
  EnvironmentStatus,
  ExportResult,
  JobStarted,
  LiveParams,
  Meter,
  ProjectListItem,
  ProjectNamePreview,
  ProjectSummary,
  Recording,
  StopRecordingResult,
  ModelArchitecture,
  TakeKind,
} from "@/types";

export const api = {
  getProjectSummary(project: string) {
    return invoke<ProjectSummary>("get_project_summary", { project });
  },

  listProjects() {
    return invoke<ProjectListItem[]>("list_projects");
  },

  previewProjectName(phrase: string, exceptProject: string | null) {
    return invoke<ProjectNamePreview>("preview_project_name", { phrase, exceptProject });
  },

  createProject(phrase: string) {
    return invoke<ProjectListItem>("create_project", { phrase });
  },

  renameProject(project: string, phrase: string) {
    return invoke<ProjectListItem>("rename_project", { project, phrase });
  },

  deleteProject(project: string) {
    return invoke<void>("delete_project", { project });
  },

  startRecording(project: string, kind: TakeKind = "wake", prompt: string | null = null) {
    return invoke<CaptureStart>("start_recording", { project, kind, prompt });
  },

  getRecordingMeter() {
    return invoke<Meter>("get_recording_meter");
  },

  stopRecording() {
    return invoke<StopRecordingResult>("stop_recording");
  },

  cancelRecording() {
    return invoke<void>("cancel_recording");
  },

  readRecordingAudio(project: string, id: string, kind: TakeKind = "wake") {
    return invoke<number[]>("read_recording_audio", { project, id, kind });
  },

  deleteRecording(project: string, id: string, kind: TakeKind = "wake") {
    return invoke<void>("delete_recording", { project, id, kind });
  },

  renameRecording(project: string, id: string, newName: string) {
    return invoke<void>("rename_recording", { project, id, newName });
  },

  getEnvironment() {
    return invoke<EnvironmentStatus>("get_environment");
  },

  startAugmentation(params: {
    project: string;
    variations: number;
    clipSeconds: number;
    backgroundDir: string | null;
    negativeDir: string | null;
    useSynthetic: boolean;
    useCorpora: boolean;
    corpusNegatives: number;
    useMined: boolean;
  }) {
    return invoke<JobStarted>("start_augmentation", {
      project: params.project,
      variations: params.variations,
      clipSeconds: params.clipSeconds,
      backgroundDir: params.backgroundDir,
      negativeDir: params.negativeDir,
      useSynthetic: params.useSynthetic,
      useCorpora: params.useCorpora,
      corpusNegatives: params.corpusNegatives,
      useMined: params.useMined,
    });
  },

  startTraining(params: {
    project: string;
    epochs: number;
    batchSize: number;
    learningRate: number;
    architecture: ModelArchitecture;
  }) {
    return invoke<JobStarted>("start_training", {
      architecture: params.architecture,
      project: params.project,
      epochs: params.epochs,
      batchSize: params.batchSize,
      learningRate: params.learningRate,
    });
  },

  startLive(params: LiveParams) {
    return invoke<JobStarted>("start_live", { ...params });
  },

  liveSnapshot(project: string) {
    return invoke<void>("live_snapshot", { project });
  },

  readLiveClip(project: string, file: string) {
    return invoke<number[]>("read_live_clip", { project, file });
  },

  saveLiveClip(project: string, file: string, kind: "wake" | "negative") {
    return invoke<Recording>("save_live_clip", { project, file, kind });
  },

  startMining(project: string) {
    return invoke<JobStarted>("start_mining", { project });
  },

  deleteMined(project: string) {
    return invoke<void>("delete_mined", { project });
  },

  startEvaluation(project: string) {
    return invoke<JobStarted>("start_evaluation", { project });
  },

  cancelJob() {
    return invoke<void>("cancel_job");
  },

  exportForEsp32(project: string, destination: string) {
    return invoke<ExportResult>("export_for_esp32", { project, destination });
  },
};

export function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message: unknown }).message);
  }
  return "Unexpected error. Check the pipeline console for details.";
}
