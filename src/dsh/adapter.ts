/**
 * DeepSeek Harness model-plane adapter.
 *
 * Harness owns the tool loop. This adapter projects that request onto the
 * OpenAI-compatible proxy already used by the OpenCode plugin, then maps the
 * SSE stream back into harness chunks. The proxy keeps talking to Cursor.
 */

import { fallbackModels } from "../cli/model-discovery.js";
import { createLogger } from "../utils/logger.js";
import { CursorHarnessError, isAbortError } from "./errors.js";
import {
  buildChatBody,
  countLiveImages,
  type MessageLike,
  type ToolSchemaLike,
} from "./messages.js";
import { chunksFromSsePayloads, readSseData, type ChunkLike } from "./sse.js";

const log = createLogger("dsh");

export const WORKSPACE_HEADER = "x-opencode-directory";
export const DEFAULT_CONTEXT_WINDOW = 200_000;
export const CURSOR_PROVIDER_ID = "cursor-acp";

export interface ModelListing {
  id: string;
  name: string;
  description?: string;
}

export interface HarnessRequest {
  provider: string;
  model: string;
  messages: readonly MessageLike[];
  system?: string;
  tools?: readonly ToolSchemaLike[];
  temperature?: number;
  maxTokens?: number;
  stop?: readonly string[];
  signal?: AbortSignal;
}

export interface ResolvedModel {
  provider: string;
  id: string;
  name: string;
  description?: string;
  inputModalities: readonly ["text"];
  context: { contextWindow: number };
}

export interface CursorHarnessAdapterOptions {
  apiKey?: string;
  workspaceDirectory?: string;
  contextWindow?: number;
  /** OpenAI-compatible root, including `/v1`. When set, the local proxy is not started. */
  baseURL?: string;
  fetch?: typeof fetch;
  discoverModels?: (apiKey: string | undefined, signal?: AbortSignal) => Promise<ModelListing[]>;
  startProxy?: (workspaceDirectory: string) => Promise<string>;
}

async function defaultDiscover(apiKey: string | undefined): Promise<ModelListing[]> {
  const { ModelDiscoveryService } = await import("../models/discovery.js");
  const service = new ModelDiscoveryService({ cacheTTL: 0 });
  const models = await service.discover(apiKey);
  return models.map((model) => ({
    id: model.id,
    name: model.name,
    description: model.description,
  }));
}

async function defaultStartProxy(workspaceDirectory: string): Promise<string> {
  const { ensureCursorProxyServer } = await import("../plugin.js");
  return ensureCursorProxyServer(workspaceDirectory);
}

export class CursorHarnessAdapter {
  private readonly apiKey?: string;
  private readonly workspaceDirectory: string;
  private readonly contextWindow: number;
  private readonly configuredBaseURL?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly discoverModels: (apiKey: string | undefined, signal?: AbortSignal) => Promise<ModelListing[]>;
  private readonly startProxy: (workspaceDirectory: string) => Promise<string>;
  private modelsCache: ModelListing[] | undefined;
  private baseURLCache: string | undefined;

  constructor(options: CursorHarnessAdapterOptions = {}) {
    const apiKey = options.apiKey?.trim();
    this.apiKey = apiKey && apiKey.length > 0 ? apiKey : undefined;
    this.workspaceDirectory = options.workspaceDirectory?.trim() || process.cwd();
    this.contextWindow = positiveInt(options.contextWindow, DEFAULT_CONTEXT_WINDOW);
    const baseURL = options.baseURL?.trim().replace(/\/$/, "");
    this.configuredBaseURL = baseURL && baseURL.length > 0 ? baseURL : undefined;
    this.fetchImpl = options.fetch ?? fetch;
    this.discoverModels = options.discoverModels ?? ((key) => defaultDiscover(key));
    this.startProxy = options.startProxy ?? defaultStartProxy;
  }

  providerInfo(provider: string): { id: string; name: string } {
    return { id: provider, name: "Cursor" };
  }

  async listModels(provider: string, signal?: AbortSignal): Promise<Array<ResolvedModel & { provider: string }>> {
    const models = await this.cachedModels(signal);
    return models.map((model) => this.describe(provider, model.id, model.name, model.description));
  }

  async resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<ResolvedModel> {
    throwIfAborted(signal);
    const models = await this.cachedModels(signal);
    throwIfAborted(signal);
    // Harness requires the returned id to equal the requested model string.
    const found = models.find((entry) => entry.id === model.trim());
    const name = found?.name || model.trim() || "Cursor";
    return this.describe(provider, model, name, found?.description);
  }

  async *stream(request: HarnessRequest, headers: Record<string, string>): AsyncGenerator<ChunkLike> {
    if (request.stop && request.stop.length > 0) {
      throw new CursorHarnessError(
        "Cursor does not support stop sequences",
        "UNSUPPORTED_OPTION",
      );
    }
    if (request.signal?.aborted) {
      yield* abortedFinish();
      return;
    }

    const liveImages = countLiveImages(request.messages);
    if (liveImages > 0) {
      throw new CursorHarnessError(
        `Cursor adapter accepts text only; offload ${liveImages} image(s) before retrying`,
        "IMAGE_OFFLOAD_REQUIRED",
        { offloadImages: liveImages },
      );
    }

    let body;
    try {
      body = buildChatBody(request);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.startsWith("UNSUPPORTED_BLOCK:")) {
        throw new CursorHarnessError(
          `Cursor adapter cannot represent content block ${message.slice("UNSUPPORTED_BLOCK:".length)}`,
          "UNSUPPORTED_OPTION",
          { cause: error },
        );
      }
      throw error;
    }

    const baseURL = await this.resolveBaseURL();
    const workspace = encodeURIComponent(this.workspaceDirectory);
    log.debug("Dispatching Cursor request", {
      provider: request.provider,
      model: body.cursorModel,
      messages: body.messages.length,
      tools: body.tools?.length ?? 0,
    });

    let response: Response;
    try {
      response = await this.fetchImpl(`${baseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...headers,
          ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
          [WORKSPACE_HEADER]: workspace,
        },
        body: JSON.stringify(body),
        signal: request.signal,
      });
    } catch (error) {
      if (request.signal?.aborted || isAbortError(error)) {
        yield* abortedFinish();
        return;
      }
      throw new CursorHarnessError(
        error instanceof Error ? error.message : String(error),
        "TRANSPORT",
        { cause: error },
      );
    }

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw failureFromHttp(response.status, text);
    }

    if (!response.body) {
      throw new CursorHarnessError("Cursor proxy returned an empty body", "CURSOR_PROXY_ERROR");
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("text/event-stream")) {
      const text = await response.text().catch(() => "");
      throw failureFromHttp(response.status, text || "Cursor proxy did not return an event stream");
    }

    yield* chunksFromSsePayloads(readSseData(response.body, request.signal), request.signal);
  }

  private describe(provider: string, id: string, name: string, description?: string): ResolvedModel {
    return {
      provider,
      id,
      name,
      ...(description ? { description } : {}),
      inputModalities: ["text"],
      context: { contextWindow: this.contextWindow },
    };
  }

  private async cachedModels(signal?: AbortSignal): Promise<ModelListing[]> {
    if (this.modelsCache) return this.modelsCache;
    throwIfAborted(signal);
    try {
      const discovered = await withAbort(startDiscovery(this.discoverModels, this.apiKey, signal), signal);
      throwIfAborted(signal);
      const models = sanitizeModels(discovered);
      if (models.length > 0) {
        this.modelsCache = models;
        return models;
      }
    } catch (error) {
      if (signal?.aborted || isAbortError(error)) throw error;
      log.debug("Model discovery failed; using static fallback", { error: String(error) });
    }
    throwIfAborted(signal);
    const fallback = sanitizeModels(fallbackModels());
    this.modelsCache = fallback;
    return fallback;
  }

  private async resolveBaseURL(): Promise<string> {
    if (this.configuredBaseURL) return this.configuredBaseURL;
    if (this.baseURLCache) return this.baseURLCache;
    const baseURL = await this.startProxy(this.workspaceDirectory);
    this.baseURLCache = baseURL.replace(/\/$/, "");
    return this.baseURLCache;
  }
}

function positiveInt(value: number | undefined, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return fallback;
  return Math.floor(value);
}

function startDiscovery(
  discover: CursorHarnessAdapterOptions["discoverModels"],
  apiKey: string | undefined,
  signal?: AbortSignal,
): Promise<ModelListing[]> {
  if (!discover) return Promise.resolve([]);
  try {
    return Promise.resolve(discover(apiKey, signal));
  } catch (error) {
    return Promise.reject(error);
  }
}

function abortError(): Error {
  const error = new Error("The Cursor request was aborted");
  error.name = "AbortError";
  return error;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError();
}

/** Settle as soon as `signal` aborts, even when `work` cannot be cancelled. */
function withAbort<T>(work: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return work;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const fail = () => reject(abortError());
    signal.addEventListener("abort", fail, { once: true });
    work.then(
      (value) => {
        signal.removeEventListener("abort", fail);
        if (signal.aborted) reject(abortError());
        else resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", fail);
        if (signal.aborted) reject(abortError());
        else reject(error);
      },
    );
  });
}

export function sanitizeModels(models: readonly ModelListing[]): ModelListing[] {
  const seen = new Set<string>();
  const clean: ModelListing[] = [];
  for (const model of models) {
    const id = typeof model?.id === "string" ? model.id.trim() : "";
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const name = typeof model?.name === "string" && model.name.trim().length > 0 ? model.name.trim() : id;
    const description = typeof model?.description === "string" && model.description.length > 0
      ? model.description
      : undefined;
    clean.push({ id, name, ...(description ? { description } : {}) });
  }
  return clean;
}

function* abortedFinish(): Generator<ChunkLike> {
  yield { type: "usage", usage: { inputTokens: 0, outputTokens: 0 } };
  yield {
    type: "finish",
    reason: {
      kind: "aborted",
      failure: { message: "The Cursor request was aborted", code: "ABORTED" },
    },
  };
}

export function failureFromHttp(status: number, body: string): CursorHarnessError {
  const message = extractErrorMessage(body) || `Cursor proxy request failed (${status})`;
  const haystack = `${status} ${message}`.toLowerCase();
  let code = "CURSOR_HTTP_ERROR";
  if (status === 401 || status === 403) code = "AUTH";
  else if (status === 429) code = "RATE_LIMIT";
  else if (/quota|insufficient credits|billing|usage limit|exceeded your/.test(haystack)) code = "QUOTA";
  else if (/context window|context length|maximum context|too many tokens|context_length/.test(haystack)) {
    code = "CONTEXT_WINDOW_EXCEEDED";
  } else if (status === 408 || status === 504) code = "TIMEOUT";
  else if (status >= 500 && status <= 599) code = "SERVER";
  return new CursorHarnessError(message, code, { status });
}

function extractErrorMessage(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed) as { error?: unknown; message?: unknown };
    if (typeof parsed.error === "string") return parsed.error;
    if (parsed.error && typeof parsed.error === "object" && "message" in parsed.error) {
      const message = (parsed.error as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
    if (typeof parsed.message === "string") return parsed.message;
  } catch {
    // Plain text error body.
  }
  return trimmed.slice(0, 500);
}
