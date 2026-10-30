import type { ProjectVoice, SyntheticSummary } from "@/lib/voice";

export type Recording = {
  id: string;
  fileName: string;
  durationMs: number;
  sizeBytes: number;
  createdAt: string;
  phrase: string;
  sourceSampleRate: number;
  trimmedMs: number;
  vadThreshold: number;
  /** Text read aloud for an "other speech" take. */
  prompt?: string;
};

export type TakeKind = "wake" | "negative";

/** Classifier head: global average pooling, or a head that keeps the order of sounds. */
export type ModelArchitecture = "ds-cnn" | "ds-cnn-temporal";

export type SplitCounts = {
  wakeWord: number;
  background: number;
};

export type DatasetSummary = {
  exists: boolean;
  train: SplitCounts;
  val: SplitCounts;
  test: SplitCounts;
  total: number;
  clipSeconds: number | null;
  variationsPerSample: number | null;
  createdAt: string | null;
  backgroundSource: string | null;
  warnings: string[];
};

export type EpochMetrics = {
  epoch: number;
  accuracy: number | null;
  loss: number | null;
  val_accuracy: number | null;
  val_loss: number | null;
  learning_rate?: number | null;
};

export type TrainingHistory = {
  epochs: EpochMetrics[];
  epochs_requested: number;
  epochs_completed: number;
  best_epoch: number;
  best_val_accuracy: number | null;
  best_val_loss?: number | null;
  selection_metric?: string;
  weights_restored?: boolean;
};

export type Quantization = {
  scale: number;
  zero_point: number;
};

export type ModelMetadata = {
  phrase: string;
  created_at: string;
  architecture: string;
  labels: string[];
  clip_seconds: number;
  sample_rate: number;
  input: {
    shape: number[];
    dtype: string;
    quantization: Quantization;
  };
  output: {
    shape: number[];
    dtype: string;
    quantization: Quantization;
  };
  preprocessing: {
    sample_rate: number;
    window_samples: number;
    hop_samples: number;
    fft_size: number;
    mel_bins: number;
    fmin_hz: number;
    fmax_hz: number;
    frame_count: number;
    logmel_mean: number;
    logmel_std: number;
  };
  metrics: {
    test_accuracy: number | null;
    test_precision: number | null;
    test_recall: number | null;
    test_f1: number | null;
    tflite_test_accuracy: number | null;
    confusion_matrix?: number[][];
  };
  training: {
    epochs_requested: number;
    epochs_completed: number;
    batch_size: number;
    learning_rate: number;
    best_epoch: number;
    best_val_accuracy: number | null;
    best_val_loss?: number | null;
    selection_metric?: string;
    dataset_counts?: Record<string, SplitCounts>;
  };
  model: {
    parameters: number;
    tflite_size_bytes: number;
    recommended_tensor_arena_bytes: number;
  };
  artifacts: Record<string, string>;
};

export type ModelSummary = {
  exists: boolean;
  metadata: ModelMetadata | null;
  history: TrainingHistory | null;
  tfliteSizeBytes: number;
  headerAvailable: boolean;
  updatedAt: string | null;
  evaluation: StreamingEvaluation | null;
};

export type OperatingPoint = {
  targetFaPerHour: number;
  /** "all:k" (k windows in a row) or "mean:k" (average of the last k scores). */
  detector: string;
  mode: "all" | "mean";
  windows: number;
  threshold: number;
  faPerHour: number;
  recall: number | null;
};

/** evaluate.py output: firmware-style streaming benchmark of the exported model. */
export type StreamingEvaluation = {
  createdAt: string;
  phrase: string;
  stepMs: number;
  refractorySeconds: number;
  speechHours: number;
  falseActivations: Array<{
    name: string;
    hours: number;
    top: Array<{ utterance: number; score: number; text: string }>;
  }>;
  recall: { takes?: string[]; trials?: number };
  /** curve[detector rule][threshold as "0.50"] */
  curve: Record<string, Record<string, { faPerHour: number; recall: number | null }>>;
  operatingPoints: OperatingPoint[];
  detector: OperatingPoint | null;
  /** Recall of each held-out take at the exported setting, weakest first. */
  perTake?: Array<{
    take: string;
    trials: number;
    recall: number;
    medianPeak: number;
    byContext: Record<string, string>;
  }>;
};

export type ProjectPaths = {
  projectDir: string;
  recordingsDir: string;
  negativesDir: string;
  datasetDir: string;
  modelsDir: string;
  exportsDir: string;
};

export type ProjectSummary = {
  phrase: string;
  slug: string;
  recordings: Recording[];
  negativeTakes: Recording[];
  /** Hard negative windows mined from training speech by an earlier model. */
  minedCount: number;
  dataset: DatasetSummary;
  model: ModelSummary;
  paths: ProjectPaths;
  voice: ProjectVoice;
  synthetic: SyntheticSummary | null;
};

export type CaptureStart = {
  id: string;
  deviceName: string;
  inputSampleRate: number;
  channels: number;
  maxSeconds: number;
  startedAt: string;
};

export type Meter = {
  levels: number[];
  rms: number;
  peak: number;
  durationMs: number;
};

export type TrimReport = {
  threshold: number;
  originalMs: number;
  speechMs: number;
  trimmedMs: number;
  trimStartMs: number;
  trimEndMs: number;
  resampledFrom: number;
};

export type StopRecordingResult = {
  recording: Recording;
  trim: TrimReport;
  captureMs: number;
  outputSampleRate: number;
};

export type EnvironmentStatus = {
  ready: boolean;
  python: string;
  executable: string;
  version: string;
  supportedPython: boolean;
  packages: Record<string, boolean>;
  message: string;
  scriptDir: string;
};

export type JobStarted = {
  jobId: string;
  project: string;
  kind: "augment" | "train" | string;
  python: string;
  script: string;
};

export type JobLogEvent = {
  jobId: string;
  project: string;
  kind: string;
  stream: "stdout" | "stderr" | string;
  line: string;
  timestamp: string;
};

export type JobProgress = {
  jobId: string;
  project: string;
  kind: string;
  stage: string;
  progress: number;
  message: string;
  epoch?: number;
  epochs?: number;
  metrics?: EpochMetrics;
};

/** Settings of the live test (python/live.py). */
export type LiveParams = {
  project: string;
  rule: "all" | "mean";
  windows: number;
  threshold: number;
  refractoryMs: number;
  gainDb: number;
};

export type LiveEvent = {
  id: number;
  kind: "detection" | "missed";
  ms: number;
  at: string;
  score?: number;
  rule?: number;
  peakDb?: number;
  file: string;
  saved: boolean;
};

/** Extra fields of the live job's progress events. */
export type LiveProgress = JobProgress & {
  score: number;
  rule: number;
  rmsDb: number;
  peakDb: number;
  ms: number;
  detections: number;
  events: LiveEvent[];
  device: string;
};

export type JobFinishedEvent = {
  jobId: string;
  project: string;
  kind: string;
  success: boolean;
  canceled: boolean;
  exitCode: number | null;
  result: Record<string, unknown> | null;
  message: string;
};

export type ProjectListItem = {
  phrase: string;
  slug: string;
  recordingCount: number;
  hasDataset: boolean;
  hasModel: boolean;
  createdAt: string | null;
  updatedAt: string | null;
};

export type ProjectNamePreview = {
  phrase: string;
  slug: string;
  error: string | null;
};

export type ExportResult = {
  destination: string;
  files: string[];
};
