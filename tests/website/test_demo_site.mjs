import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_DEMO_LIMITS,
  buildDefaultRequestId,
  getJobUiState,
  parseHintPhrases,
  shouldContinuePolling,
  validateUploadMetadata,
} from "../../website/assets/app.mjs";

test("parseHintPhrases trims, deduplicates, and caps the public list", () => {
  const hints = parseHintPhrases(" wallet, floral | wallet\neditorial, leather, studio, hero, carry, campaign, extra ");
  assert.deepEqual(hints, [
    "wallet",
    "floral",
    "editorial",
    "leather",
    "studio",
    "hero",
    "carry",
    "campaign",
  ]);
});

test("buildDefaultRequestId creates a deterministic browser-safe slug", () => {
  const requestId = buildDefaultRequestId(
    "Floral Wallet",
    new Date("2026-04-18T12:34:56.000Z")
  );
  assert.equal(requestId, "floral-wallet-20260418123456");
});

test("validateUploadMetadata rejects unsupported mime types and oversize files", () => {
  assert.equal(
    validateUploadMetadata({ type: "image/gif", size: 1234 }, DEFAULT_DEMO_LIMITS).ok,
    false
  );
  assert.equal(
    validateUploadMetadata(
      { type: "image/png", size: DEFAULT_DEMO_LIMITS.maxSourceUploadBytes + 1 },
      DEFAULT_DEMO_LIMITS
    ).ok,
    false
  );
  assert.equal(
    validateUploadMetadata({ type: "image/png", size: 1234 }, DEFAULT_DEMO_LIMITS).ok,
    true
  );
});

test("getJobUiState maps the user-visible job phases", () => {
  assert.equal(getJobUiState("queued").headline, "Waiting for a GPU worker");
  assert.equal(getJobUiState("running").label, "Running");
  assert.equal(getJobUiState("succeeded").tone, "tone-succeeded");
  assert.equal(getJobUiState("invalid_source").tone, "tone-invalid");
  assert.equal(getJobUiState("failed").tone, "tone-failed");
});

test("shouldContinuePolling only for non-terminal broker statuses", () => {
  assert.equal(shouldContinuePolling("queued"), true);
  assert.equal(shouldContinuePolling("running"), true);
  assert.equal(shouldContinuePolling("succeeded"), false);
  assert.equal(shouldContinuePolling("invalid_source"), false);
  assert.equal(shouldContinuePolling("failed"), false);
});
