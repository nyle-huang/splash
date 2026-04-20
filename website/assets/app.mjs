export const DEFAULT_DEMO_LIMITS = Object.freeze({
  maxUploadBytes: 4_500_000,
  maxSourceUploadBytes: 12_000_000,
  targetMaxEncodedBytes: 3_200_000,
  targetMaxDimension: 1600,
  pollIntervalMs: 3500,
});

const SUPPORTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_HINT_PHRASES = 8;

export function normalizeRequestId(value) {
  if (!value) {
    return "";
  }
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

export function buildDefaultRequestId(productTitle, now = new Date()) {
  const prefix = normalizeRequestId(productTitle).slice(0, 28) || "campaign-demo";
  const stamp = now.toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
  return `${prefix}-${stamp}`;
}

export function parseHintPhrases(value) {
  const items = String(value || "")
    .split(/[\n,|]/g)
    .map((item) => item.trim())
    .filter(Boolean);
  return [...new Set(items)].slice(0, MAX_HINT_PHRASES);
}

export function validateUploadMetadata(file, limits = DEFAULT_DEMO_LIMITS) {
  if (!file) {
    return { ok: false, message: "Choose an image before starting a request." };
  }
  if (!SUPPORTED_MIME_TYPES.has(file.type)) {
    return { ok: false, message: "Only JPEG, PNG, and WebP uploads are supported." };
  }
  if (file.size <= 0) {
    return { ok: false, message: "The selected file is empty." };
  }
  if (file.size > limits.maxSourceUploadBytes) {
    return {
      ok: false,
      message: `The selected file is too large to compress safely in the browser (${formatBytes(file.size)}).`,
    };
  }
  return { ok: true, message: "The file can be prepared for upload." };
}

export function shouldContinuePolling(status) {
  return status === "queued" || status === "running";
}

export function getJobUiState(status) {
  switch (status) {
    case "queued":
      return {
        tone: "tone-queued",
        label: "Queued",
        headline: "Waiting for a GPU worker",
      };
    case "running":
      return {
        tone: "tone-running",
        label: "Running",
        headline: "Generating the campaign image",
      };
    case "succeeded":
      return {
        tone: "tone-succeeded",
        label: "Succeeded",
        headline: "Final image ready",
      };
    case "invalid_source":
      return {
        tone: "tone-invalid",
        label: "Invalid source",
        headline: "The upload failed input-quality checks",
      };
    default:
      return {
        tone: "tone-failed",
        label: "Failed",
        headline: "The request did not complete",
      };
  }
}

function resolveConfig() {
  return { ...DEFAULT_DEMO_LIMITS, ...(window.PRODUCT_CAMPAIGN_DEMO_CONFIG || {}) };
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function requireConfiguredBrokerUrl(config) {
  const value = String(config.brokerBaseUrl || "").trim();
  if (!value || value.includes("YOUR-AZURE-FUNCTION-APP")) {
    throw new Error(
      "Set website/assets/config.js with the Azure Function base URL before using the public site."
    );
  }
  return value.replace(/\/+$/g, "");
}

function readBlobAsDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read the prepared image."));
    reader.readAsDataURL(blob);
  });
}

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("The selected file could not be decoded as an image."));
    };
    image.src = url;
  });
}

function resizeDimensions(width, height, maxDimension) {
  if (Math.max(width, height) <= maxDimension) {
    return { width, height };
  }
  const scale = maxDimension / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

function drawResizedImage(image, dimensions) {
  const canvas = document.createElement("canvas");
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    throw new Error("The browser could not create a 2D canvas context.");
  }
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function canvasToJpegBlob(canvas, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("The browser could not encode the prepared image."));
          return;
        }
        resolve(blob);
      },
      "image/jpeg",
      quality
    );
  });
}

async function encodePreparedImage(canvas, limits) {
  let workingCanvas = canvas;
  let quality = 0.92;
  let blob = await canvasToJpegBlob(workingCanvas, quality);

  while (blob.size > limits.targetMaxEncodedBytes && quality > 0.62) {
    quality = Number((quality - 0.08).toFixed(2));
    blob = await canvasToJpegBlob(workingCanvas, quality);
  }

  while (blob.size > limits.targetMaxEncodedBytes && Math.max(workingCanvas.width, workingCanvas.height) > 960) {
    const nextDimensions = {
      width: Math.max(1, Math.round(workingCanvas.width * 0.86)),
      height: Math.max(1, Math.round(workingCanvas.height * 0.86)),
    };
    const smallerCanvas = document.createElement("canvas");
    smallerCanvas.width = nextDimensions.width;
    smallerCanvas.height = nextDimensions.height;
    const context = smallerCanvas.getContext("2d", { alpha: false });
    if (!context) {
      throw new Error("The browser could not resize the prepared image.");
    }
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, nextDimensions.width, nextDimensions.height);
    context.drawImage(workingCanvas, 0, 0, nextDimensions.width, nextDimensions.height);
    workingCanvas = smallerCanvas;
    blob = await canvasToJpegBlob(workingCanvas, quality);
  }

  if (blob.size > limits.targetMaxEncodedBytes) {
    throw new Error(
      `The prepared image is still too large after compression (${formatBytes(blob.size)}).`
    );
  }

  const dataUrl = await readBlobAsDataUrl(blob);
  return {
    blob,
    dataUrl,
    imageBase64: dataUrl.split(",", 2)[1] || "",
    mimeType: "image/jpeg",
    width: workingCanvas.width,
    height: workingCanvas.height,
  };
}

async function prepareUploadFromFile(file, limits) {
  const validation = validateUploadMetadata(file, limits);
  if (!validation.ok) {
    throw new Error(validation.message);
  }
  const image = await loadImageElement(file);
  const dimensions = resizeDimensions(image.naturalWidth, image.naturalHeight, limits.targetMaxDimension);
  const canvas = drawResizedImage(image, dimensions);
  const prepared = await encodePreparedImage(canvas, limits);
  return {
    ...prepared,
    originalMimeType: file.type,
    originalBytes: file.size,
    originalWidth: image.naturalWidth,
    originalHeight: image.naturalHeight,
    processedBytes: prepared.blob.size,
  };
}

function detailFromResponse(payload) {
  if (!payload || payload.detail === undefined || payload.detail === null) {
    return "";
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || item.type || "request error").join("; ");
  }
  return String(payload.detail);
}

function buildBrokerRequest(form, preparedUpload) {
  return {
    image_base64: preparedUpload.imageBase64,
    mime_type: preparedUpload.mimeType,
    product_title: form.productTitle.value.trim(),
    hint_phrases: parseHintPhrases(form.hintPhrases.value),
    request_id: normalizeRequestId(form.requestId.value) || buildDefaultRequestId(form.productTitle.value),
  };
}

function setTextContent(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function toggleHidden(element, hidden) {
  if (element) {
    element.classList.toggle("is-hidden", hidden);
  }
}

function buildTraceLine(message) {
  const line = document.createElement("div");
  line.className = "trace-item";

  const dot = document.createElement("div");
  dot.className = "trace-dot";

  const text = document.createElement("div");
  text.textContent = message;

  line.append(dot, text);
  return line;
}

function createDomBindings() {
  return {
    form: document.querySelector("#demo-form"),
    token: document.querySelector("#demo-token"),
    productTitle: document.querySelector("#product-title"),
    hintPhrases: document.querySelector("#hint-phrases"),
    requestId: document.querySelector("#request-id"),
    fileInput: document.querySelector("#source-image"),
    submitButton: document.querySelector("#submit-button"),
    resetButton: document.querySelector("#reset-button"),
    previewImage: document.querySelector("#upload-preview-image"),
    previewEmpty: document.querySelector("#upload-preview-empty"),
    previewShell: document.querySelector("#upload-preview-shell"),
    originalBytes: document.querySelector("#original-bytes"),
    preparedBytes: document.querySelector("#prepared-bytes"),
    preparedDimensions: document.querySelector("#prepared-dimensions"),
    mimeLabel: document.querySelector("#prepared-mime"),
    statusPill: document.querySelector("#status-pill"),
    statusHeadline: document.querySelector("#status-headline"),
    statusSummary: document.querySelector("#status-summary"),
    statusTrace: document.querySelector("#status-trace"),
    resultImage: document.querySelector("#result-image"),
    resultFrame: document.querySelector("#result-frame"),
    resultMeta: document.querySelector("#result-meta"),
    invalidCallout: document.querySelector("#invalid-callout"),
    invalidReason: document.querySelector("#invalid-reason"),
    invalidIssues: document.querySelector("#invalid-issues"),
    errorCallout: document.querySelector("#error-callout"),
    errorText: document.querySelector("#error-text"),
  };
}

function updatePreparedUploadPreview(bindings, preparedUpload) {
  bindings.previewImage.src = preparedUpload.dataUrl;
  bindings.previewImage.alt = "Prepared upload preview";
  setTextContent(bindings.originalBytes, formatBytes(preparedUpload.originalBytes));
  setTextContent(bindings.preparedBytes, formatBytes(preparedUpload.processedBytes));
  setTextContent(
    bindings.preparedDimensions,
    `${preparedUpload.width} × ${preparedUpload.height}px`
  );
  setTextContent(bindings.mimeLabel, preparedUpload.mimeType);
  toggleHidden(bindings.previewShell, false);
  toggleHidden(bindings.previewEmpty, true);
}

function renderJobState(bindings, response) {
  const uiState = getJobUiState(response.status);
  bindings.statusPill.className = `status-pill ${uiState.tone}`;
  setTextContent(bindings.statusPill, uiState.label);
  setTextContent(bindings.statusHeadline, uiState.headline);
  setTextContent(bindings.statusSummary, response.summary || "");
  bindings.statusTrace.replaceChildren(
    buildTraceLine(`Job id: ${response.job_id || "pending"}`),
    buildTraceLine(`Broker status: ${response.status}`),
    buildTraceLine(
      response.selected_candidate_mode
        ? `Selected campaign mode: ${response.selected_candidate_mode}`
        : "Selected campaign mode: pending"
    )
  );

  if (response.final_image_base64 && response.final_image_mime_type) {
    bindings.resultImage.src = `data:${response.final_image_mime_type};base64,${response.final_image_base64}`;
    bindings.resultImage.alt = "Generated campaign result";
    toggleHidden(bindings.resultFrame, false);
    toggleHidden(bindings.resultMeta, false);
    setTextContent(
      bindings.resultMeta,
      response.selected_candidate_mode
        ? `Selected campaign mode: ${response.selected_candidate_mode}`
        : "The worker returned a final image without a selected candidate mode."
    );
  } else {
    bindings.resultImage.removeAttribute("src");
    toggleHidden(bindings.resultFrame, response.status !== "running" && response.status !== "queued");
    toggleHidden(bindings.resultMeta, true);
  }

  if (response.invalid_source) {
    setTextContent(bindings.invalidReason, response.invalid_source.reason || "invalid_source_photo");
    bindings.invalidIssues.replaceChildren(
      ...((response.invalid_source.issues || []).map((issue) => {
        const item = document.createElement("li");
        item.textContent = issue;
        return item;
      }))
    );
    toggleHidden(bindings.invalidCallout, false);
  } else {
    bindings.invalidIssues.replaceChildren();
    toggleHidden(bindings.invalidCallout, true);
  }

  if (response.status === "failed") {
    setTextContent(
      bindings.errorText,
      response.error_code
        ? `${response.summary} Error code: ${response.error_code}.`
        : response.summary
    );
    toggleHidden(bindings.errorCallout, false);
  } else {
    setTextContent(bindings.errorText, "");
    toggleHidden(bindings.errorCallout, true);
  }
}

async function postJson(url, token, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(detailFromResponse(body) || `Request failed with status ${response.status}.`);
  }
  return body;
}

async function getJson(url, token) {
  const response = await fetch(url, {
    method: "GET",
    headers: {
      authorization: `Bearer ${token}`,
    },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(detailFromResponse(body) || `Request failed with status ${response.status}.`);
  }
  return body;
}

async function pollUntilComplete({ baseUrl, jobId, token, bindings, config }) {
  let latest = null;
  for (;;) {
    latest = await getJson(`${baseUrl}/api/jobs/${encodeURIComponent(jobId)}`, token);
    renderJobState(bindings, latest);
    if (!shouldContinuePolling(latest.status)) {
      return latest;
    }
    await new Promise((resolve) => window.setTimeout(resolve, config.pollIntervalMs));
  }
}

function resetStatus(bindings) {
  renderJobState(bindings, {
    status: "queued",
    job_id: "not-started",
    summary: "Submit a prepared upload to create a GPU job.",
  });
  bindings.resultImage.removeAttribute("src");
  toggleHidden(bindings.resultMeta, true);
  toggleHidden(bindings.invalidCallout, true);
  toggleHidden(bindings.errorCallout, true);
}

function bootDemo() {
  const config = resolveConfig();
  const bindings = createDomBindings();
  if (!bindings.form) {
    return;
  }

  const state = {
    preparedUpload: null,
    activeJobId: "",
    submitting: false,
  };

  resetStatus(bindings);

  bindings.fileInput.addEventListener("change", async () => {
    try {
      state.preparedUpload = null;
      const file = bindings.fileInput.files?.[0];
      if (!file) {
        toggleHidden(bindings.previewShell, true);
        toggleHidden(bindings.previewEmpty, false);
        return;
      }
      const preparedUpload = await prepareUploadFromFile(file, config);
      state.preparedUpload = preparedUpload;
      updatePreparedUploadPreview(bindings, preparedUpload);
    } catch (error) {
      state.preparedUpload = null;
      setTextContent(bindings.errorText, error instanceof Error ? error.message : String(error));
      toggleHidden(bindings.errorCallout, false);
      toggleHidden(bindings.previewShell, true);
      toggleHidden(bindings.previewEmpty, false);
    }
  });

  bindings.resetButton.addEventListener("click", () => {
    bindings.form.reset();
    state.preparedUpload = null;
    state.activeJobId = "";
    toggleHidden(bindings.previewShell, true);
    toggleHidden(bindings.previewEmpty, false);
    resetStatus(bindings);
  });

  bindings.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.submitting) {
      return;
    }

    try {
      const token = bindings.token.value.trim();
      const productTitle = bindings.productTitle.value.trim();
      if (!token) {
        throw new Error("Enter the demo token that you plan to share privately with reviewers.");
      }
      if (!productTitle) {
        throw new Error("Enter a product title before starting a request.");
      }
      if (!state.preparedUpload) {
        throw new Error("Choose a supported image and wait for browser-side preparation to finish.");
      }

      state.submitting = true;
      bindings.submitButton.disabled = true;
      bindings.submitButton.textContent = "Submitting…";
      toggleHidden(bindings.errorCallout, true);

      const baseUrl = requireConfiguredBrokerUrl(config);
      const payload = buildBrokerRequest(bindings, state.preparedUpload);
      const queued = await postJson(`${baseUrl}/api/jobs`, token, payload);
      state.activeJobId = queued.job_id;
      renderJobState(bindings, queued);
      const finalResponse = await pollUntilComplete({
        baseUrl,
        jobId: queued.job_id,
        token,
        bindings,
        config,
      });
      if (!shouldContinuePolling(finalResponse.status)) {
        bindings.submitButton.textContent = "Submit another request";
      }
    } catch (error) {
      setTextContent(bindings.errorText, error instanceof Error ? error.message : String(error));
      toggleHidden(bindings.errorCallout, false);
      bindings.submitButton.textContent = "Retry request";
      state.activeJobId = "";
    } finally {
      state.submitting = false;
      bindings.submitButton.disabled = false;
    }
  });
}

if (typeof window !== "undefined" && window.document) {
  window.addEventListener("DOMContentLoaded", bootDemo, { once: true });
}
