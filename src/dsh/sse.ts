/**
 * Translate the local Cursor proxy's OpenAI SSE stream into DeepSeek Harness
 * chunks. Usage is emitted immediately before the terminal finish.
 */

import { CursorHarnessError, isAbortError } from "./errors.js";

export interface TokenUsageLike {
  inputTokens: number;
  outputTokens: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
}

export type FinishLike =
  | { kind: "stop" }
  | { kind: "tool-calls" }
  | { kind: "max-tokens" }
  | { kind: "aborted"; failure: { message: string; code: string } };

export type ChunkLike =
  | { type: "block-start"; index: number; blockType: "text" | "reasoning" | "tool-call" }
  | { type: "text-delta"; index: number; text: string }
  | { type: "reasoning-delta"; index: number; text: string }
  | { type: "tool-call-delta"; index: number; id: string; name?: string; argumentsDelta: string }
  | { type: "block-end"; index: number; block: Record<string, unknown> }
  | { type: "usage"; usage: TokenUsageLike }
  | { type: "finish"; reason: FinishLike };

interface ToolAcc {
  openaiIndex: number;
  blockIndex: number;
  id: string;
  name: string;
  args: string;
}

interface OpenBlock {
  kind: "text" | "reasoning";
  index: number;
  text: string;
}

export function usageFromOpenAi(raw: unknown): TokenUsageLike | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const usage = raw as Record<string, unknown>;
  const prompt = readCount(usage.prompt_tokens);
  const completion = readCount(usage.completion_tokens);
  const details = isRecord(usage.prompt_tokens_details) ? usage.prompt_tokens_details : {};
  const completionDetails = isRecord(usage.completion_tokens_details) ? usage.completion_tokens_details : {};
  const cacheRead = readCount(details.cached_tokens);
  const cacheWrite = readCount(details.cache_write_tokens);
  const reasoning = readCount(completionDetails.reasoning_tokens);
  const inputTokens = Math.max(0, prompt - cacheRead - cacheWrite);
  const hasAny = prompt > 0 || completion > 0 || cacheRead > 0 || cacheWrite > 0 || reasoning > 0;
  if (!hasAny) return undefined;
  const mapped: TokenUsageLike = {
    inputTokens,
    outputTokens: completion,
  };
  if (typeof usage.total_tokens === "number" && Number.isFinite(usage.total_tokens)) {
    mapped.totalTokens = Math.max(0, Math.floor(usage.total_tokens));
  }
  if (cacheRead > 0) mapped.cacheReadTokens = cacheRead;
  if (cacheWrite > 0) mapped.cacheWriteTokens = cacheWrite;
  if (reasoning > 0) mapped.reasoningTokens = reasoning;
  return mapped;
}

function readCount(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return 0;
  return Math.floor(value);
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object";
}

/**
 * Incremental assembler. `push` yields deltas as they arrive; `finish` closes
 * blocks, then usage, then the terminal chunk.
 */
export class OpenAiChunkAssembler {
  private nextIndex = 0;
  private text: OpenBlock | null = null;
  private reasoning: OpenBlock | null = null;
  private readonly tools: ToolAcc[] = [];
  private finishReason: string | null = null;
  private usage: TokenUsageLike | undefined;
  private sawContent = false;

  get hasContent(): boolean {
    return this.sawContent;
  }

  *push(payload: unknown): Generator<ChunkLike> {
    if (!isRecord(payload)) return;
    if (payload.error !== undefined && !Array.isArray(payload.choices)) {
      const message = typeof payload.error === "string"
        ? payload.error
        : (isRecord(payload.error) && typeof payload.error.message === "string"
          ? payload.error.message
          : "Cursor proxy returned an error");
      throw new CursorHarnessError(message, "CURSOR_PROXY_ERROR");
    }

    if (payload.usage !== undefined) {
      const usage = usageFromOpenAi(payload.usage);
      if (usage) this.usage = usage;
    }

    const choice = Array.isArray(payload.choices) ? payload.choices[0] : undefined;
    if (!isRecord(choice)) return;
    if (typeof choice.finish_reason === "string" && choice.finish_reason.length > 0) {
      this.finishReason = choice.finish_reason;
    }
    const delta = isRecord(choice.delta) ? choice.delta : {};

    if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0) {
      yield* this.reasoningDelta(delta.reasoning_content);
    }
    if (typeof delta.content === "string" && delta.content.length > 0) {
      yield* this.textDelta(delta.content);
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        yield* this.toolDelta(call);
      }
    }
  }

  *finish(options?: { aborted?: boolean }): Generator<ChunkLike> {
    if (!options?.aborted && !this.sawContent) {
      throw new CursorHarnessError("Cursor returned an empty response", "EMPTY_RESPONSE");
    }

    const blocks = [
      ...(this.reasoning ? [this.reasoning] : []),
      ...(this.text ? [this.text] : []),
    ].sort((a, b) => a.index - b.index);

    for (const block of blocks) {
      yield {
        type: "block-end",
        index: block.index,
        block: { type: block.kind, text: block.text },
      };
    }

    const tools = [...this.tools].sort((a, b) => a.blockIndex - b.blockIndex);
    for (const tool of tools) {
      yield {
        type: "block-end",
        index: tool.blockIndex,
        block: {
          type: "tool-call",
          id: tool.id,
          name: tool.name || "tool",
          arguments: tool.args || "{}",
        },
      };
    }

    const usage = this.usage ?? { inputTokens: 0, outputTokens: 0 };
    yield { type: "usage", usage };

    if (options?.aborted) {
      yield {
        type: "finish",
        reason: {
          kind: "aborted",
          failure: { message: "The Cursor request was aborted", code: "ABORTED" },
        },
      };
      return;
    }

    yield { type: "finish", reason: this.finishKind(tools.length > 0) };
  }

  private finishKind(hasTools: boolean): FinishLike {
    if (hasTools || this.finishReason === "tool_calls") return { kind: "tool-calls" };
    if (this.finishReason === "length" || this.finishReason === "max_tokens") return { kind: "max-tokens" };
    return { kind: "stop" };
  }

  private *textDelta(text: string): Generator<ChunkLike> {
    if (!this.text) {
      this.text = { kind: "text", index: this.nextIndex++, text: "" };
      yield { type: "block-start", index: this.text.index, blockType: "text" };
    }
    this.text.text += text;
    this.sawContent = true;
    yield { type: "text-delta", index: this.text.index, text };
  }

  private *reasoningDelta(text: string): Generator<ChunkLike> {
    if (!this.reasoning) {
      this.reasoning = { kind: "reasoning", index: this.nextIndex++, text: "" };
      yield { type: "block-start", index: this.reasoning.index, blockType: "reasoning" };
    }
    this.reasoning.text += text;
    this.sawContent = true;
    yield { type: "reasoning-delta", index: this.reasoning.index, text };
  }

  private *toolDelta(call: unknown): Generator<ChunkLike> {
    if (!isRecord(call)) return;
    const openaiIndex = typeof call.index === "number" ? call.index : this.tools.length;
    let acc = this.tools.find((tool) => tool.openaiIndex === openaiIndex);
    const fn = isRecord(call.function) ? call.function : {};
    const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
    const name = typeof fn.name === "string" && fn.name.length > 0 ? fn.name : undefined;
    const args = typeof fn.arguments === "string" ? fn.arguments : "";

    if (!acc) {
      acc = {
        openaiIndex,
        blockIndex: this.nextIndex++,
        id: id || `call_${openaiIndex}`,
        name: name || "",
        args: "",
      };
      this.tools.push(acc);
      this.sawContent = true;
      yield { type: "block-start", index: acc.blockIndex, blockType: "tool-call" };
    } else {
      if (id) acc.id = id;
      if (name) acc.name = name;
    }

    if (args.length > 0) acc.args += args;
    this.sawContent = true;
    yield {
      type: "tool-call-delta",
      index: acc.blockIndex,
      id: acc.id,
      ...(acc.name ? { name: acc.name } : {}),
      argumentsDelta: args,
    };
  }
}

/** Split an SSE byte stream into `data:` payloads, excluding the framing. */
export async function* readSseData(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let sep = buffer.indexOf("\n\n");
      while (sep >= 0) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const data = sseData(raw);
        if (data !== undefined) yield data;
        sep = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    const tail = sseData(buffer);
    if (tail !== undefined) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function sseData(raw: string): string | undefined {
  const lines = raw.split("\n").map((line) => line.trimEnd()).filter((line) => line.startsWith("data:"));
  if (lines.length === 0) return undefined;
  return lines.map((line) => line.slice(5).trimStart()).join("\n");
}

export async function* chunksFromSsePayloads(
  payloads: AsyncIterable<string>,
  signal?: AbortSignal,
): AsyncGenerator<ChunkLike> {
  const assembler = new OpenAiChunkAssembler();
  try {
    for await (const data of payloads) {
      if (signal?.aborted) {
        yield* assembler.finish({ aborted: true });
        return;
      }
      if (data === "[DONE]") break;
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch (error) {
        throw new CursorHarnessError("Cursor proxy sent malformed SSE", "CURSOR_PROXY_ERROR", { cause: error });
      }
      yield* assembler.push(parsed);
    }
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) {
      yield* assembler.finish({ aborted: true });
      return;
    }
    throw error;
  }
  if (signal?.aborted) {
    yield* assembler.finish({ aborted: true });
    return;
  }
  yield* assembler.finish();
}
