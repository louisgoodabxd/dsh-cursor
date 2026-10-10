/**
 * DeepSeek Harness bundle entry.
 *
 * Install with `dsh plugin add` (see docs/deepseek-harness.md). The patch
 * loads this subpath, registers the `cursor-acp` provider, and leaves the
 * package root export for OpenCode.
 */
import type { Context } from "@deepseek-ai/cordis";
import {
  LlmAdapter,
  LlmError,
  attributionHeaders,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmResolvedModelInfo,
  type StreamChunk,
} from "@deepseek-ai/dsh-llm";
import z from "@deepseek-ai/schemastery";
import { setStoredApiKey } from "./plugin.js";
import { CursorHarnessAdapter, CURSOR_PROVIDER_ID, type HarnessRequest } from "./dsh/adapter.js";
import { CursorHarnessError } from "./dsh/errors.js";

export const name = "llm-cursor";
export const inject = ["llm"] as const;

export interface Config {
  apiKey?: string;
  workspaceDirectory?: string;
  providers?: string[];
  contextWindow?: number;
  baseURL?: string;
}

export const Config: z<Config> = z.object({
  apiKey: z.string(),
  workspaceDirectory: z.string(),
  providers: z.array(z.string()),
  contextWindow: z.number(),
  baseURL: z.string(),
});

class CursorDshAdapter extends LlmAdapter {
  constructor(private readonly core: CursorHarnessAdapter) {
    super();
  }

  override providerInfo(provider: string) {
    return this.core.providerInfo(provider);
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return this.core.listModels(provider);
  }

  override resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    return this.core.resolveModel(provider, model, signal);
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    try {
      const chunks = this.core.stream(options as unknown as HarnessRequest, attributionHeaders());
      for await (const chunk of chunks) {
        yield chunk as StreamChunk;
      }
    } catch (error) {
      throw asLlmError(error);
    }
  }
}

export function apply(ctx: Context, config: Config = {}): void {
  const apiKey = firstNonEmpty(config.apiKey, process.env.CURSOR_API_KEY);
  if (apiKey) setStoredApiKey(apiKey);

  const providers = (config.providers ?? []).map((provider) => provider.trim()).filter((provider) => provider.length > 0);
  const routes = [...new Set(providers.length > 0 ? providers : [CURSOR_PROVIDER_ID])];
  const core = new CursorHarnessAdapter({
    apiKey,
    workspaceDirectory: firstNonEmpty(config.workspaceDirectory, process.cwd()),
    contextWindow: config.contextWindow,
    baseURL: config.baseURL,
  });
  const adapter = new CursorDshAdapter(core);

  const registerDirectory = (ctx.llm as { registerConfigurableProviders?: (entries: unknown[]) => void })
    .registerConfigurableProviders;
  if (typeof registerDirectory === "function") {
    registerDirectory.call(ctx.llm, routes.map((provider) => ({
      provider,
      displayName: "Cursor",
      settingsNs: "llm-cursor",
      settingsPath: [] as const,
    })));
  }

  ctx.llm.registerAdapter(routes, adapter);
}

function firstNonEmpty(...values: Array<string | undefined>): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) return value.trim();
  }
  return undefined;
}

export function asLlmError(error: unknown): LlmError {
  if (error instanceof LlmError) return error;
  if (error instanceof CursorHarnessError) {
    return new LlmError(error.message, error.code, {
      ...(error.status !== undefined ? { status: error.status } : {}),
      ...(error.offloadImages !== undefined ? { offloadImages: error.offloadImages } : {}),
      cause: error,
    });
  }
  const message = error instanceof Error ? error.message : String(error);
  return new LlmError(message || "Cursor request failed", "CURSOR_PROXY_ERROR", {
    cause: error instanceof Error ? error : undefined,
  });
}
