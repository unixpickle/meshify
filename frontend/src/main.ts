import { createRun, listRuns, waitForRunsEvent } from "./api.js";
import type { Asset, Run, Stage } from "./types.js";

const rootElement = document.querySelector<HTMLDivElement>("#app");

if (!rootElement) {
  throw new Error("Missing app container");
}

const app: HTMLDivElement = rootElement;

type ViewerState = {
  cameraOrbit?: string;
  cameraTarget?: string;
  fieldOfView?: string;
};

type ModelViewerElement = HTMLElement & {
  src?: string;
  cameraOrbit?: string;
  cameraTarget?: string;
  fieldOfView?: string;
  getCameraOrbit?: () => { toString: () => string };
  getCameraTarget?: () => { toString: () => string };
  getFieldOfView?: () => { toString: () => string };
};

type RouteState = {
  runId: string;
  homeScrollY: number;
};

const state = {
  runs: [] as Run[],
  eventId: 0,
  selectedRunId: "",
  uploading: false,
  uploadError: "",
  selectedFile: null as File | null,
  viewerState: {} as Record<string, ViewerState>,
  lastDetailRunId: "",
  lastStageSignature: "",
  lastAssetPanelSignature: "",
  homeScrollY: 0,
};

let pollGeneration = 0;
let pollController: AbortController | null = null;

const refs = {
  homeView: null as HTMLElement | null,
  totalRuns: null as HTMLSpanElement | null,
  runningRuns: null as HTMLSpanElement | null,
  completedRuns: null as HTMLSpanElement | null,
  uploadForm: null as HTMLFormElement | null,
  fileInput: null as HTMLInputElement | null,
  filePickerName: null as HTMLSpanElement | null,
  submitButton: null as HTMLButtonElement | null,
  refreshButton: null as HTMLButtonElement | null,
  uploadError: null as HTMLSpanElement | null,
  runList: null as HTMLDivElement | null,
  detailHost: null as HTMLElement | null,
};

function formatTime(value: string | null): string {
  if (!value) return "Not started";
  return new Date(value).toLocaleString();
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function stageClass(stage: Stage, currentStage: string): string {
  if (stage.status === "failed") return "stage-card failed";
  if (stage.status === "completed") return "stage-card done";
  if (stage.stage_key === currentStage && stage.status === "running") return "stage-card active";
  return "stage-card";
}

function groupAssets(run: Run): Map<string, Asset[]> {
  const groups = new Map<string, Asset[]>();
  for (const asset of run.assets) {
    const current = groups.get(asset.stage_key) ?? [];
    current.push(asset);
    groups.set(asset.stage_key, current);
  }
  return groups;
}

function buildStageSignature(run: Run): string {
  return run.stages
    .map((stage) => [stage.stage_key, stage.status, stage.progress, stage.message ?? ""].join("|"))
    .join("||");
}

function buildAssetPanelSignature(run: Run): string {
  const assetsByStage = groupAssets(run);
  return run.stages
    .filter((stage) => assetsByStage.has(stage.stage_key))
    .map((stage) => {
      const assets = assetsByStage.get(stage.stage_key) ?? [];
      const assetSignature = assets
        .map((asset) => [asset.id, asset.kind, asset.label, asset.url].join("|"))
        .join("~~");
      return [stage.stage_key, stage.status, stage.message ?? "", assetSignature].join("::");
    })
    .join("||");
}

function getSelectedRun(): Run | undefined {
  return state.runs.find((run) => run.id === state.selectedRunId);
}

function applyRunsSnapshot(runs: Run[]): void {
  state.runs = runs;
  if (state.selectedRunId && !runs.some((run) => run.id === state.selectedRunId)) {
    state.selectedRunId = "";
    resetDetailCache();
    updateLocation("", false);
  }
  syncUI();
}

function upsertRun(run: Run): void {
  const nextRuns = state.runs.filter((item) => item.id !== run.id);
  nextRuns.unshift(run);
  applyRunsSnapshot(nextRuns);
}

function resetDetailCache(): void {
  state.lastDetailRunId = "";
  state.lastStageSignature = "";
  state.lastAssetPanelSignature = "";
}

function currentRunIdFromLocation(): string {
  const match = window.location.hash.match(/^#run\/([A-Za-z0-9_-]+)$/);
  return match?.[1] ?? "";
}

function readRouteState(): RouteState {
  const current = window.history.state as Partial<RouteState> | null;
  return {
    runId: typeof current?.runId === "string" ? current.runId : "",
    homeScrollY: typeof current?.homeScrollY === "number" ? current.homeScrollY : state.homeScrollY,
  };
}

function writeScrollPosition(top: number): void {
  window.requestAnimationFrame(() => {
    window.scrollTo({ top, left: 0, behavior: "auto" });
  });
}

function updateLocation(runId: string, push: boolean): void {
  const nextUrl = runId ? `${window.location.pathname}#run/${runId}` : window.location.pathname;
  const method = push ? "pushState" : "replaceState";
  window.history[method]({ runId, homeScrollY: state.homeScrollY }, "", nextUrl);
}

function navigateToRun(runId: string, push = true): void {
  if (!state.selectedRunId) {
    state.homeScrollY = window.scrollY;
    updateLocation("", false);
  }
  if (state.selectedRunId === runId) {
    if (push) {
      updateLocation(runId, true);
    }
    syncUI();
    writeScrollPosition(0);
    return;
  }
  state.selectedRunId = runId;
  resetDetailCache();
  if (push) {
    updateLocation(runId, true);
  }
  syncUI();
  writeScrollPosition(0);
}

function navigateHome(push = true): void {
  if (!state.selectedRunId && !push) {
    syncUI();
    writeScrollPosition(state.homeScrollY);
    return;
  }
  state.selectedRunId = "";
  resetDetailCache();
  updateLocation("", push);
  syncUI();
  writeScrollPosition(state.homeScrollY);
}

function createShell(): void {
  app.innerHTML = `
    <main class="shell">
      <section class="home-view" data-role="home-view">
        <section class="hero">
          <article class="hero-card">
            <span class="eyebrow">Single User Mesh Pipeline</span>
            <h1>Meshify Control Room</h1>
            <p>Queue a new image, then open any run to inspect its stage timeline and generated outputs.</p>
            <div class="stats">
              <div class="stat">
                <span class="stat-label">Total Runs</span>
                <span class="stat-value" data-role="total-runs">0</span>
              </div>
              <div class="stat">
                <span class="stat-label">Running Now</span>
                <span class="stat-value" data-role="running-runs">0</span>
              </div>
              <div class="stat">
                <span class="stat-label">Completed</span>
                <span class="stat-value" data-role="completed-runs">0</span>
              </div>
            </div>
          </article>
        </section>

        <section class="home-stack">
          <section class="panel">
            <h2>Queue New Image</h2>
            <p class="muted">Upload a source image to start a fresh pipeline run.</p>
            <form class="upload-form" id="upload-form">
              <input class="file-input" type="file" id="file-input" accept="image/*" />
              <label class="file-picker" for="file-input">
                <span class="file-picker-button" data-role="file-picker-button">Browse</span>
                <span class="file-picker-name" data-role="file-picker-name">Choose an image</span>
              </label>
              <div class="upload-actions">
                <button class="primary" type="submit" data-role="submit-button">Start Pipeline</button>
                <button class="secondary" type="button" data-role="refresh-button">Refresh</button>
              </div>
              <span class="muted" data-role="upload-error"></span>
            </form>
          </section>

          <section class="panel">
            <h2>Models</h2>
            <p class="muted">Every run is listed here. Click one to open its stage details and outputs.</p>
            <div class="run-list" data-role="run-list"></div>
          </section>
        </section>
      </section>

      <section class="detail-view" data-role="detail-host" hidden></section>
    </main>
  `;

  refs.homeView = app.querySelector('[data-role="home-view"]');
  refs.totalRuns = app.querySelector('[data-role="total-runs"]');
  refs.runningRuns = app.querySelector('[data-role="running-runs"]');
  refs.completedRuns = app.querySelector('[data-role="completed-runs"]');
  refs.uploadForm = app.querySelector("#upload-form");
  refs.fileInput = app.querySelector("#file-input");
  refs.filePickerName = app.querySelector('[data-role="file-picker-name"]');
  refs.submitButton = app.querySelector('[data-role="submit-button"]');
  refs.refreshButton = app.querySelector('[data-role="refresh-button"]');
  refs.uploadError = app.querySelector('[data-role="upload-error"]');
  refs.runList = app.querySelector('[data-role="run-list"]');
  refs.detailHost = app.querySelector('[data-role="detail-host"]');

  refs.uploadForm?.addEventListener("submit", onUploadSubmit);
  refs.fileInput?.addEventListener("change", onFileInputChange);
  refs.refreshButton?.addEventListener("click", () => {
    void reloadRunsNow();
    restartLongPolling();
  });
}

function syncSummary(): void {
  if (!refs.totalRuns || !refs.runningRuns || !refs.completedRuns) return;
  refs.totalRuns.textContent = `${state.runs.length}`;
  refs.runningRuns.textContent = `${state.runs.filter((run) => run.status === "running").length}`;
  refs.completedRuns.textContent = `${state.runs.filter((run) => run.status === "completed").length}`;
}

function syncViewMode(): void {
  const showingDetail = Boolean(state.selectedRunId);
  if (refs.homeView) {
    refs.homeView.hidden = showingDetail;
  }
  if (refs.detailHost) {
    refs.detailHost.hidden = !showingDetail;
  }
}

function syncUploadForm(): void {
  if (!refs.fileInput || !refs.filePickerName || !refs.submitButton || !refs.uploadError) return;
  refs.fileInput.disabled = state.uploading;
  refs.filePickerName.textContent = state.selectedFile?.name ?? "Choose an image";
  refs.submitButton.disabled = state.uploading;
  refs.submitButton.textContent = state.uploading ? "Uploading..." : "Start Pipeline";
  refs.uploadError.textContent = state.uploadError;
}

function createRunCard(run: Run): HTMLElement {
  const element = document.createElement("article");
  element.className = "run-card";
  element.dataset.runId = run.id;
  element.innerHTML = `
    <div class="run-thumb"></div>
    <div class="run-body">
      <div class="run-topline">
        <span class="pill" data-role="status-pill"></span>
        <span class="muted" data-role="created-at"></span>
      </div>
      <strong data-role="name"></strong>
      <span class="muted" data-role="message"></span>
      <div class="progress-shell">
        <div class="progress-bar" data-role="progress-bar"></div>
      </div>
    </div>
  `;
  element.addEventListener("click", () => {
    navigateToRun(run.id);
  });
  return element;
}

function placeChild(container: HTMLElement, child: HTMLElement, index: number): void {
  const currentAtIndex = container.children.item(index);
  if (currentAtIndex === child) {
    return;
  }
  container.insertBefore(child, currentAtIndex);
}

function updateRunCard(element: HTMLElement, run: Run): void {
  element.classList.toggle("selected", run.id === state.selectedRunId);
  const thumb = element.querySelector<HTMLElement>(".run-thumb");
  if (thumb) {
    const existingImage = thumb.querySelector<HTMLImageElement>("img");
    if (run.preview_image_url) {
      if (existingImage) {
        if (existingImage.src !== new URL(run.preview_image_url, window.location.origin).toString()) {
          existingImage.src = run.preview_image_url;
        }
        existingImage.alt = run.original_name;
      } else {
        thumb.innerHTML = `<img src="${run.preview_image_url}" alt="${escapeHtml(run.original_name)}" />`;
      }
    } else {
      thumb.innerHTML = "";
    }
  }

  const pill = element.querySelector<HTMLElement>('[data-role="status-pill"]');
  if (pill) {
    pill.className = `pill ${run.status}`;
    pill.textContent = run.status;
  }
  const createdAt = element.querySelector<HTMLElement>('[data-role="created-at"]');
  if (createdAt) createdAt.textContent = formatTime(run.created_at);
  const name = element.querySelector<HTMLElement>('[data-role="name"]');
  if (name) name.textContent = run.original_name;
  const message = element.querySelector<HTMLElement>('[data-role="message"]');
  if (message) message.textContent = run.message ?? "Queued";
  const progressBar = element.querySelector<HTMLElement>('[data-role="progress-bar"]');
  if (progressBar) progressBar.style.width = formatPercent(run.progress);
}

function syncRunList(): void {
  if (!refs.runList) return;
  const container = refs.runList;
  const existing = new Map<string, HTMLElement>();
  container.querySelectorAll<HTMLElement>(".run-card[data-run-id]").forEach((element) => {
    existing.set(element.dataset.runId ?? "", element);
  });

  if (state.runs.length === 0) {
    container.innerHTML = `
      <section class="empty-state">
        <strong>No uploads yet.</strong>
        <p class="muted">The global run history will appear here once the first image is submitted.</p>
      </section>
    `;
    return;
  }

  container.querySelector(".empty-state")?.remove();

  for (const run of state.runs) {
    const index = state.runs.indexOf(run);
    let element = existing.get(run.id);
    if (!element) {
      element = createRunCard(run);
    }
    updateRunCard(element, run);
    placeChild(container, element, index);
    existing.delete(run.id);
  }

  for (const stale of existing.values()) {
    stale.remove();
  }
}

function createDetailShell(): void {
  if (!refs.detailHost) return;
  refs.detailHost.innerHTML = `
    <section class="detail-view-shell">
      <div class="detail-nav">
        <button class="secondary" type="button" data-role="back-button">Back to Home</button>
      </div>

      <section class="detail-grid" data-role="detail-grid">
      <header class="detail-header" data-role="detail-header">
        <div class="detail-topline">
          <span class="pill" data-role="detail-status"></span>
          <span class="muted" data-role="detail-created"></span>
          <span class="muted" data-role="detail-updated"></span>
        </div>
        <h2 data-role="detail-name"></h2>
        <p class="muted" data-role="detail-message"></p>
        <div class="progress-shell">
          <div class="progress-bar" data-role="detail-progress"></div>
        </div>
      </header>

      <section class="detail-columns">
        <div class="panel">
          <h2>Pipeline Stages</h2>
          <p class="muted">Live stage state is read from SQLite-backed progress records.</p>
          <div class="stage-list" data-role="stage-list"></div>
        </div>
        <div class="panel">
          <h2>Stage Outputs</h2>
          <p class="muted">Image stages show their generated images, and the export stage exposes the GLB in a 3D viewer.</p>
          <div data-role="asset-groups-host"></div>
        </div>
      </section>
      </section>
    </section>
  `;
  refs.detailHost.querySelector<HTMLButtonElement>('[data-role="back-button"]')?.addEventListener("click", () => {
    navigateHome(false);
  });
}

function syncEmptyDetail(): void {
  if (!refs.detailHost) return;
  refs.detailHost.innerHTML = `
    <section class="detail-view-shell">
      <div class="detail-nav">
        <button class="secondary" type="button" data-role="back-button">Back to Home</button>
      </div>
      <section class="empty-state">
        <strong>Run not found.</strong>
        <p class="muted">Go back to the homepage and choose another model from the list.</p>
      </section>
    </section>
  `;
  refs.detailHost.querySelector<HTMLButtonElement>('[data-role="back-button"]')?.addEventListener("click", () => {
    navigateHome(false);
  });
}

function createStageCard(stage: Stage, currentStage: string): HTMLElement {
  const element = document.createElement("article");
  element.dataset.stageKey = stage.stage_key;
  element.innerHTML = `
    <div class="meta-row">
      <strong data-role="label"></strong>
      <span class="pill" data-role="status"></span>
    </div>
    <p class="muted" data-role="message"></p>
    <div class="progress-shell">
      <div class="progress-bar" data-role="progress"></div>
    </div>
  `;
  updateStageCard(element, stage, currentStage);
  return element;
}

function updateStageCard(element: HTMLElement, stage: Stage, currentStage: string): void {
  element.className = stageClass(stage, currentStage);
  const label = element.querySelector<HTMLElement>('[data-role="label"]');
  if (label) label.textContent = stage.stage_label;
  const status = element.querySelector<HTMLElement>('[data-role="status"]');
  if (status) {
    status.className = `pill ${stage.status}`;
    status.textContent = stage.status;
  }
  const message = element.querySelector<HTMLElement>('[data-role="message"]');
  if (message) message.textContent = stage.message ?? "Waiting";
  const progress = element.querySelector<HTMLElement>('[data-role="progress"]');
  if (progress) progress.style.width = formatPercent(stage.progress);
}

function syncStages(run: Run): void {
  const container = refs.detailHost?.querySelector<HTMLElement>('[data-role="stage-list"]');
  if (!container) return;

  const existing = new Map<string, HTMLElement>();
  container.querySelectorAll<HTMLElement>("[data-stage-key]").forEach((element) => {
    existing.set(element.dataset.stageKey ?? "", element);
  });

  for (const stage of run.stages) {
    const index = run.stages.indexOf(stage);
    let element = existing.get(stage.stage_key);
    if (!element) {
      element = createStageCard(stage, run.current_stage);
    }
    updateStageCard(element, stage, run.current_stage);
    placeChild(container, element, index);
    existing.delete(stage.stage_key);
  }

  for (const stale of existing.values()) {
    stale.remove();
  }
}

function attachViewerPersistence(viewer: ModelViewerElement, assetId: string): void {
  if (viewer.dataset.boundCameraState === "true") {
    return;
  }
  viewer.dataset.boundCameraState = "true";
  viewer.addEventListener("camera-change", () => {
    state.viewerState[assetId] = {
      cameraOrbit: viewer.getCameraOrbit?.().toString(),
      cameraTarget: viewer.getCameraTarget?.().toString(),
      fieldOfView: viewer.getFieldOfView?.().toString(),
    };
  });
}

function createAssetCard(asset: Asset): HTMLElement {
  const element = document.createElement("article");
  element.className = "asset-card";
  element.dataset.assetId = asset.id;
  element.dataset.assetKind = asset.kind;

  const mediaHost = document.createElement("div");
  mediaHost.dataset.role = "media-host";
  element.appendChild(mediaHost);

  const meta = document.createElement("div");
  meta.className = "asset-meta";
  meta.innerHTML = `
    <strong data-role="label"></strong>
    <span class="muted" data-role="stage"></span>
    <div class="asset-actions" data-role="actions"></div>
  `;
  element.appendChild(meta);

  updateAssetCard(element, asset);
  return element;
}

function updateAssetCard(element: HTMLElement, asset: Asset): void {
  element.dataset.assetId = asset.id;
  const mediaHost = element.querySelector<HTMLElement>('[data-role="media-host"]');
  if (!mediaHost) return;

  if (asset.kind === "model") {
    let viewer = mediaHost.querySelector<ModelViewerElement>("model-viewer");
    if (!viewer) {
      viewer = document.createElement("model-viewer") as ModelViewerElement;
      mediaHost.replaceChildren(viewer);
      attachViewerPersistence(viewer, asset.id);
      const viewerState = state.viewerState[asset.id];
      viewer.dataset.assetId = asset.id;
      viewer.setAttribute("src", asset.url);
      viewer.setAttribute("camera-controls", "");
      viewer.setAttribute("camera-orbit", viewerState?.cameraOrbit ?? "45deg 70deg auto");
      viewer.setAttribute("camera-target", viewerState?.cameraTarget ?? "auto auto auto");
      viewer.setAttribute("field-of-view", viewerState?.fieldOfView ?? "28deg");
      viewer.setAttribute("shadow-intensity", "0.36");
      viewer.setAttribute("exposure", "0.48");
      viewer.setAttribute("interaction-prompt", "none");
      viewer.setAttribute("poster", "");
    } else if ((viewer.getAttribute("src") ?? "") !== asset.url) {
      const viewerState = state.viewerState[asset.id];
      viewer.dataset.assetId = asset.id;
      viewer.setAttribute("src", asset.url);
      viewer.setAttribute("shadow-intensity", "0.36");
      viewer.setAttribute("exposure", "0.48");
      if (viewerState?.cameraOrbit) {
        viewer.setAttribute("camera-orbit", viewerState.cameraOrbit);
      }
      if (viewerState?.cameraTarget) {
        viewer.setAttribute("camera-target", viewerState.cameraTarget);
      }
      if (viewerState?.fieldOfView) {
        viewer.setAttribute("field-of-view", viewerState.fieldOfView);
      }
    }
  } else {
    let image = mediaHost.querySelector<HTMLImageElement>("img");
    if (!image) {
      image = document.createElement("img");
      mediaHost.replaceChildren(image);
    }
    image.src = asset.url;
    image.alt = asset.label;
  }

  const label = element.querySelector<HTMLElement>('[data-role="label"]');
  if (label) label.textContent = asset.label;
  const stage = element.querySelector<HTMLElement>('[data-role="stage"]');
  if (stage) stage.textContent = asset.stage_key;
  const actions = element.querySelector<HTMLElement>('[data-role="actions"]');
  if (actions) {
    if (asset.kind === "model") {
      actions.innerHTML = `<a class="download-link" href="${asset.url}" download>Download GLB</a>`;
    } else {
      actions.replaceChildren();
    }
  }
}

function createAssetGroup(stage: Stage): HTMLElement {
  const element = document.createElement("section");
  element.className = "asset-group";
  element.dataset.stageKey = stage.stage_key;
  element.innerHTML = `
    <div>
      <h3 data-role="group-title"></h3>
      <p class="muted" data-role="group-message"></p>
    </div>
    <div class="asset-grid" data-role="asset-grid"></div>
  `;
  return element;
}

function syncAssetGroups(run: Run): void {
  const host = refs.detailHost?.querySelector<HTMLElement>('[data-role="asset-groups-host"]');
  if (!host) return;

  const assetsByStage = groupAssets(run);
  const stagesWithAssets = run.stages.filter((stage) => assetsByStage.has(stage.stage_key));

  if (stagesWithAssets.length === 0) {
    host.innerHTML = `
      <section class="empty-state">
        <strong>No stage assets yet.</strong>
        <p class="muted">As the pipeline runs, processed images and the final 3D mesh will appear here.</p>
      </section>
    `;
    return;
  }

  host.querySelector(".empty-state")?.remove();

  const existingGroups = new Map<string, HTMLElement>();
  const duplicateGroups: HTMLElement[] = [];
  host.querySelectorAll<HTMLElement>("[data-stage-key]").forEach((element) => {
    const stageKey = element.dataset.stageKey ?? "";
    if (!stageKey) {
      return;
    }
    if (existingGroups.has(stageKey)) {
      duplicateGroups.push(element);
      return;
    }
    existingGroups.set(stageKey, element);
  });
  for (const duplicate of duplicateGroups) {
    duplicate.remove();
  }

  for (const stage of stagesWithAssets) {
    const groupIndex = stagesWithAssets.indexOf(stage);
    const assets = assetsByStage.get(stage.stage_key) ?? [];
    let group = existingGroups.get(stage.stage_key);
    if (!group) {
      group = createAssetGroup(stage);
    }

    const title = group.querySelector<HTMLElement>('[data-role="group-title"]');
    if (title) title.textContent = stage.stage_label;
    const message = group.querySelector<HTMLElement>('[data-role="group-message"]');
    if (message) message.textContent = stage.message ?? "Stage output";

    const grid = group.querySelector<HTMLElement>('[data-role="asset-grid"]');
    if (grid) {
      const existingAssets = new Map<string, HTMLElement>();
      const duplicateAssets: HTMLElement[] = [];
      grid.querySelectorAll<HTMLElement>("[data-asset-id]").forEach((element) => {
        const assetId = element.dataset.assetId ?? "";
        if (!assetId) {
          return;
        }
        if (existingAssets.has(assetId)) {
          duplicateAssets.push(element);
          return;
        }
        existingAssets.set(assetId, element);
      });
      for (const duplicate of duplicateAssets) {
        duplicate.remove();
      }

      for (const asset of assets) {
        const assetIndex = assets.indexOf(asset);
        let card = existingAssets.get(asset.id);
        if (!card) {
          card = createAssetCard(asset);
        }
        updateAssetCard(card, asset);
        placeChild(grid, card, assetIndex);
        existingAssets.delete(asset.id);
      }

      for (const stale of existingAssets.values()) {
        stale.remove();
      }
    }

    placeChild(host, group, groupIndex);
    existingGroups.delete(stage.stage_key);
  }

  for (const stale of existingGroups.values()) {
    stale.remove();
  }
}

function syncDetail(): void {
  const run = getSelectedRun();
  if (!run) {
    state.lastDetailRunId = "";
    state.lastStageSignature = "";
    state.lastAssetPanelSignature = "";
    syncEmptyDetail();
    return;
  }

  if (!refs.detailHost?.querySelector('[data-role="detail-grid"]')) {
    createDetailShell();
  }

  const detailStatus = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-status"]');
  if (detailStatus) {
    detailStatus.className = `pill ${run.status}`;
    detailStatus.textContent = run.status;
  }

  const detailCreated = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-created"]');
  if (detailCreated) detailCreated.textContent = `Created ${formatTime(run.created_at)}`;
  const detailUpdated = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-updated"]');
  if (detailUpdated) detailUpdated.textContent = `Updated ${formatTime(run.updated_at)}`;
  const detailName = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-name"]');
  if (detailName) detailName.textContent = run.original_name;
  const detailMessage = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-message"]');
  if (detailMessage) detailMessage.textContent = run.message ?? "Queued";
  const detailProgress = refs.detailHost?.querySelector<HTMLElement>('[data-role="detail-progress"]');
  if (detailProgress) detailProgress.style.width = formatPercent(run.progress);

  const detailRunChanged = state.lastDetailRunId !== run.id;
  const stageSignature = buildStageSignature(run);
  if (detailRunChanged || stageSignature !== state.lastStageSignature) {
    syncStages(run);
    state.lastStageSignature = stageSignature;
  }

  const assetPanelSignature = buildAssetPanelSignature(run);
  if (detailRunChanged || assetPanelSignature !== state.lastAssetPanelSignature) {
    syncAssetGroups(run);
    state.lastAssetPanelSignature = assetPanelSignature;
  }

  state.lastDetailRunId = run.id;
}

function syncUI(): void {
  syncViewMode();
  syncSummary();
  syncUploadForm();
  syncRunList();
  if (state.selectedRunId) {
    syncDetail();
  }
}

async function reloadRunsNow(): Promise<void> {
  applyRunsSnapshot(await listRuns());
}

async function onUploadSubmit(event: Event): Promise<void> {
  event.preventDefault();
  const file = state.selectedFile;
  if (!file) {
    state.uploadError = "Choose an image first.";
    syncUI();
    return;
  }

  state.uploading = true;
  state.uploadError = "";
  syncUI();

  try {
    upsertRun(await createRun(file));
    state.selectedFile = null;
    if (refs.fileInput) {
      refs.fileInput.value = "";
    }
  } catch (error) {
    state.uploadError = error instanceof Error ? error.message : "Upload failed";
    syncUI();
  } finally {
    state.uploading = false;
    syncUI();
  }
}

function onFileInputChange(event: Event): void {
  const input = event.currentTarget as HTMLInputElement | null;
  state.selectedFile = input?.files?.[0] ?? null;
  state.uploadError = "";
  syncUploadForm();
}

function syncRouteFromLocation(): void {
  const routeState = readRouteState();
  state.homeScrollY = routeState.homeScrollY;
  state.selectedRunId = currentRunIdFromLocation();
  resetDetailCache();
  syncUI();
  writeScrollPosition(state.selectedRunId ? 0 : state.homeScrollY);
}

async function pollLoop(generation: number): Promise<void> {
  while (generation === pollGeneration) {
    const controller = new AbortController();
    pollController = controller;
    const timeoutId = window.setTimeout(() => {
      controller.abort();
    }, 35000);

    try {
      const event = await waitForRunsEvent(state.eventId, controller.signal);
      window.clearTimeout(timeoutId);
      if (generation !== pollGeneration) {
        return;
      }
      state.eventId = event.event_id;
      if (event.runs) {
        applyRunsSnapshot(event.runs);
      }
    } catch (error) {
      window.clearTimeout(timeoutId);
      if (generation !== pollGeneration) {
        return;
      }
      await delay(document.visibilityState === "hidden" ? 2500 : 1000);
      try {
        await reloadRunsNow();
      } catch {
        // Keep the reconnect loop alive even if a fallback snapshot fails.
      }
    } finally {
      if (pollController === controller) {
        pollController = null;
      }
    }
  }
}

function restartLongPolling(): void {
  pollGeneration += 1;
  pollController?.abort();
  void pollLoop(pollGeneration);
}

async function boot(): Promise<void> {
  createShell();
  window.history.scrollRestoration = "manual";
  state.homeScrollY = readRouteState().homeScrollY;
  state.selectedRunId = currentRunIdFromLocation();
  updateLocation(state.selectedRunId, false);
  syncUI();
  window.addEventListener("popstate", () => {
    syncRouteFromLocation();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      restartLongPolling();
    }
  });
  window.addEventListener("focus", () => {
    restartLongPolling();
  });
  window.addEventListener("online", () => {
    restartLongPolling();
  });

  try {
    const initialEvent = await waitForRunsEvent(0);
    state.eventId = initialEvent.event_id;
    applyRunsSnapshot(initialEvent.runs ?? []);
  } catch {
    await reloadRunsNow();
  }
  restartLongPolling();
}

void boot();
