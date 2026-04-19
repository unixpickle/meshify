export interface Stage {
  id: number;
  run_id: string;
  stage_key: string;
  stage_label: string;
  stage_order: number;
  status: "pending" | "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string | null;
  started_at: string | null;
  updated_at: string;
  completed_at: string | null;
}

export interface Asset {
  id: string;
  run_id: string;
  stage_key: string;
  kind: "image" | "model";
  label: string;
  storage_path: string;
  mime_type: string;
  created_at: string;
  metadata: Record<string, unknown>;
  url: string;
}

export interface Run {
  id: string;
  original_name: string;
  status: "queued" | "running" | "completed" | "failed";
  current_stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  settings: Record<string, unknown>;
  stages: Stage[];
  assets: Asset[];
  preview_image_url: string | null;
  final_model_url: string | null;
}

export interface RunsEvent {
  event_id: number;
  changed: boolean;
  event: {
    event_id: number;
    kind: string;
    run_id: string | null;
    created_at: string;
  } | null;
  runs?: Run[];
}
