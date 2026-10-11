import { describe, expect, test } from "bun:test";
import { Context } from "@deepseek-ai/cordis";
import { BlockAssembler, LlmError, LlmRuntime } from "@deepseek-ai/dsh-llm";
import { createOpenAiUsage } from "../../src/usage.js";
import { OPENCODE_DIRECTORY_HEADER, setStoredApiKey } from "../../src/plugin.js";
import { CursorHarnessAdapter, WORKSPACE_HEADER } from "../../src/dsh/adapter.js";
import { CursorHarnessError } from "../../src/dsh/errors.js";
import { buildChatBody, cursorModelId } from "../../src/dsh/messages.js";
import { chunksFromSsePayloads, OpenAiChunkAssembler, usageFromOpenAi } from "../../src/dsh/sse.js";
import type { ChunkLike } from "../../src/dsh/sse.js";
import { apply, asLlmError, Config, CursorDshAdapter, name, SETTINGS_NS } from "../../src/dsh-plugin.js";

async function collect(source: AsyncIterable<ChunkLike>): Promise<ChunkLike[]> {
  const chunks: ChunkLike[] = [];
  for await (const chunk of source) chunks.push(chunk);
  return chunks;
}

function sseBody(events: unknown[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n";
}

describe("dsh messages", () => {
  test("strips the cursor-acp prefix and projects tool history", () => {
    expect(cursorModelId("cursor-acp/composer-2")).toBe("composer-2");
    expect(cursorModelId("")).toBe("auto");
    expect(cursorModelId("auto ")).toBe("auto");

    const body = buildChatBody({
      model: "composer-2",
      system: "You are a coding agent.",
      maxTokens: 128,
      temperature: 0.2,
      tools: [
        { name: "shell", description: "Run a command", parameters: { type: "object" } },
        { name: "hidden", description: "deferred", parameters: {}, deferLoading: true },
      ],
      messages: [
        { role: "user", content: [{ type: "text", text: "List files" }] },
        {
          role: "assistant",
          content: [
            { type: "reasoning", text: "use shell" },
            { type: "text", text: "Running ls" },
            { type: "tool-call", id: "call_1", name: "shell", arguments: "{\"command\":\"ls\"}" },
          ],
        },
        {
          role: "tool",
          toolCallId: "call_1",
          isError: true,
          content: [{ type: "text", text: "denied" }],
        },
        {
          role: "developer",
          content: [{ type: "tool-addition", toolName: "read" }],
        },
        {
          role: "user",
          content: [
            { type: "file", attachment: { name: "loose.txt", bytes: 2 } },
            { type: "file", attachment: { name: "notes.txt", bytes: 4, attachmentId: "sha256:abcdef0123456789" } },
            { type: "image", offloaded: true, attachment: { name: "shot.png" } },
            {
              type: "image",
              offloaded: true,
              attachment: { name: "shot.png", attachmentId: "sha256:0123456789abcdef" },
            },
          ],
        },
      ],
    });

    expect(body.model).toBe("cursor-acp/composer-2");
    expect(body.cursorModel).toBe("composer-2");
    expect(body.stream).toBe(true);
    expect(body.max_tokens).toBe(128);
    expect(body.temperature).toBe(0.2);
    expect(body.tools?.map((tool) => (tool.function as { name: string }).name)).toEqual(["shell"]);
    expect(body.messages.map((message) => message.role)).toEqual([
      "system",
      "system",
      "user",
      "assistant",
      "tool",
      "developer",
      "user",
    ]);
    expect(body.messages[1]?.content).toBe("Keep the visible reply within 128 tokens.");
    const assistant = body.messages[3] as { content: string; tool_calls: Array<{ id: string }> };
    expect(assistant.content).toBe("Running ls");
    expect(assistant.tool_calls[0]?.id).toBe("call_1");
    expect(body.messages[4]?.content).toBe("ERROR: denied");
    expect(body.messages[5]?.content).toBe("Tool added: read");
    const projected = String(body.messages[6]?.content);
    expect(projected).toContain("[file loose.txt, 2 bytes]");
    expect(projected).toContain('File "notes.txt" (4 bytes, sha256:abcdef01)');
    expect(projected).toContain("do not claim to have read it");
    expect(projected).toContain("[image omitted to fit request image limits; shot.png.]");
    expect(projected).toContain('image omitted to fit request image limits; "shot.png" (sha256:0123456789abcdef)');
  });
});

describe("dsh sse", () => {
  test("maps text, reasoning, tool calls, and disjoint usage", () => {
    const usage = createOpenAiUsage({
      inputTokens: 10,
      outputTokens: 4,
      reasoningTokens: 2,
      cacheReadTokens: 3,
      cacheWriteTokens: 1,
    });
    const assembler = new OpenAiChunkAssembler();
    const chunks = [
      ...assembler.push({
        choices: [{ delta: { reasoning_content: "Plan" }, finish_reason: null }],
      }),
      ...assembler.push({
        choices: [{ delta: { content: "Hello" }, finish_reason: null }],
      }),
      ...assembler.push({
        choices: [{
          delta: {
            tool_calls: [{
              index: 0,
              id: "call_9",
              function: { name: "shell", arguments: "{\"command\":" },
            }],
          },
          finish_reason: null,
        }],
      }),
      ...assembler.push({
        choices: [{
          delta: {
            tool_calls: [{ index: 0, function: { arguments: "\"ls\"}" } }],
          },
          finish_reason: "tool_calls",
        }],
        usage,
      }),
      ...assembler.finish(),
    ];

    expect(chunks.map((chunk) => chunk.type)).toEqual([
      "block-start",
      "reasoning-delta",
      "block-start",
      "text-delta",
      "block-start",
      "tool-call-delta",
      "tool-call-delta",
      "block-end",
      "block-end",
      "block-end",
      "usage",
      "finish",
    ]);
    const toolEnd = chunks.filter((chunk) => chunk.type === "block-end").at(-1);
    expect(toolEnd).toMatchObject({
      block: { type: "tool-call", id: "call_9", name: "shell", arguments: "{\"command\":\"ls\"}" },
    });
    expect(usageFromOpenAi(usage)).toEqual({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: usage.total_tokens,
      cacheReadTokens: 3,
      cacheWriteTokens: 1,
      reasoningTokens: 2,
    });
    expect(chunks.at(-1)).toEqual({ type: "finish", reason: { kind: "tool-calls" } });
    expect(chunks.at(-2)?.type).toBe("usage");

    const assembled = new BlockAssembler();
    for (const chunk of chunks) assembled.push(chunk as never);
    expect(assembled.finish).toEqual({ kind: "tool-calls" });
    expect(assembled.blocks().map((block) => block.type)).toEqual(["reasoning", "text", "tool-call"]);
    expect(assembled.blocks().at(-1)).toMatchObject({
      type: "tool-call",
      id: "call_9",
      name: "shell",
      arguments: "{\"command\":\"ls\"}",
    });
  });

  test("rejects an empty stream before emitting chunks", () => {
    const assembler = new OpenAiChunkAssembler();
    expect(() => [...assembler.finish()]).toThrow(CursorHarnessError);
    try {
      [...assembler.finish()];
    } catch (error) {
      expect(error).toBeInstanceOf(CursorHarnessError);
      expect((error as CursorHarnessError).code).toBe("EMPTY_RESPONSE");
    }
  });
});

describe("dsh adapter", () => {
  test("workspace header matches the OpenCode proxy", () => {
    expect(WORKSPACE_HEADER).toBe(OPENCODE_DIRECTORY_HEADER);
  });

  test("streams a proxy SSE response and forwards harness headers", async () => {
    const requests: Array<{ url: string; headers: Headers; body: any }> = [];
    const adapter = new CursorHarnessAdapter({
      apiKey: "key_123",
      workspaceDirectory: "/work/app",
      contextWindow: 4096,
      baseURL: "http://127.0.0.1:9/v1",
      discoverModels: async () => [{ id: "auto", name: "Auto" }],
      fetch: async (input, init) => {
        requests.push({
          url: String(input),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)),
        });
        return new Response(sseBody([
          { choices: [{ delta: { content: "Hi" }, finish_reason: null }] },
          { choices: [{ delta: {}, finish_reason: "length" }], usage: { prompt_tokens: 2, completion_tokens: 1 } },
        ]), {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      },
    });

    const models = await adapter.listModels("cursor-acp");
    expect(models[0]).toMatchObject({
      provider: "cursor-acp",
      id: "auto",
      name: "Auto",
      inputModalities: ["text"],
      context: { contextWindow: 4096 },
    });
    const resolved = await adapter.resolveModel("cursor-acp", "missing-model");
    expect(resolved.id).toBe("missing-model");
    expect(resolved.name).toBe("missing-model");
    const preserved = await adapter.resolveModel("cursor-acp", "auto ");
    expect(preserved.id).toBe("auto ");
    expect(preserved.name).toBe("Auto");

    const chunks = await collect(adapter.stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
      tools: [{ name: "shell", description: "run", parameters: { type: "object" } }],
    }, { "user-agent": "deepseek-harness/test (+https://example.test)" }));

    expect(requests[0]?.url).toBe("http://127.0.0.1:9/v1/chat/completions");
    expect(requests[0]?.headers.get("authorization")).toBe("Bearer key_123");
    expect(requests[0]?.headers.get("user-agent")).toContain("deepseek-harness");
    expect(requests[0]?.headers.get(WORKSPACE_HEADER)).toBe(encodeURIComponent("/work/app"));
    expect(requests[0]?.body.cursorModel).toBe("auto");
    expect(requests[0]?.body.tools[0].function.name).toBe("shell");
    expect(chunks.at(-1)).toEqual({ type: "finish", reason: { kind: "max-tokens" } });
    expect(chunks.some((chunk) => chunk.type === "text-delta" && chunk.text === "Hi")).toBe(true);
  });

  test("rejects stop sequences, live images, and proxy auth failures", async () => {
    let calls = 0;
    const adapter = new CursorHarnessAdapter({
      baseURL: "http://127.0.0.1:9/v1",
      fetch: async () => {
        calls += 1;
        return new Response(JSON.stringify({ error: "unauthorized" }), {
          status: 401,
          headers: { "content-type": "application/json" },
        });
      },
    });

    await expectReject(adapter.stream({
      provider: "cursor-acp",
      model: "auto",
      stop: ["END"],
      messages: [],
    }, {}), "UNSUPPORTED_OPTION");
    expect(calls).toBe(0);

    const images = await expectReject(adapter.stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [
        { role: "user", content: [{ type: "image", attachment: { attachmentId: "a" } }] },
        { role: "user", content: [{ type: "image", attachment: { attachmentId: "b" } }] },
      ],
    }, {}), "IMAGE_OFFLOAD_REQUIRED");
    expect(images.offloadImages).toBe(2);
    expect(calls).toBe(0);

    const error = await expectReject(adapter.stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
    }, {}), "AUTH");
    expect(error.status).toBe(401);
    expect(calls).toBe(1);

    const server = await expectReject(new CursorHarnessAdapter({
      baseURL: "http://127.0.0.1:9/v1",
      fetch: async () => new Response("upstream down", { status: 503 }),
    }).stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
    }, {}), "SERVER");
    expect(server.status).toBe(503);

    await expectReject(new CursorHarnessAdapter({
      baseURL: "http://127.0.0.1:9/v1",
      fetch: async () => {
        throw new TypeError("fetch failed");
      },
    }).stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
    }, {}), "TRANSPORT");
  });

  test("drops blank and duplicate model ids before Harness validates the catalog", async () => {
    const adapter = new CursorHarnessAdapter({
      baseURL: "http://127.0.0.1:9/v1",
      discoverModels: async () => [
        { id: "auto", name: "  " },
        { id: "auto", name: "Auto again" },
        { id: " ", name: "blank" },
        { id: "composer-2", name: "Composer 2", description: "fast" },
      ],
    });
    const models = await adapter.listModels("cursor-acp");
    expect(models.map((model) => model.id)).toEqual(["auto", "composer-2"]);
    expect(models[0]?.name).toBe("auto");
    expect(models[1]).toMatchObject({ name: "Composer 2", description: "fast" });
  });

  test("model lookup settles when discovery is aborted", async () => {
    const controller = new AbortController();
    const adapter = new CursorHarnessAdapter({
      discoverModels: () => new Promise(() => {}),
    });
    const pending = adapter.resolveModel("cursor-acp", "auto", controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  test("a key that cannot ride in an HTTP header is refused before registration", () => {
    expect(() => apply({} as never, { apiKey: "bad key" })).toThrow(LlmError);
  });

  test("Harness rewrites images and files before the adapter, then assembles the reply", async () => {
    const bodies: Array<{ messages: Array<{ content?: string }>; tools?: Array<{ function: { name: string } }> }> = [];
    const ctx = new Context();
    await ctx.plugin(LlmRuntime);
    const core = new CursorHarnessAdapter({
      apiKey: "test-key",
      baseURL: "http://127.0.0.1:9/v1",
      discoverModels: async () => [{ id: "auto", name: "Auto" }],
      fetch: async (_input, init) => {
        bodies.push(JSON.parse(String(init?.body)));
        return new Response(sseBody([
          { choices: [{ delta: { content: "Done" }, finish_reason: null }] },
          { choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 1 } },
        ]), {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      },
    });
    ctx.llm.registerAdapter(["cursor-acp"], new CursorDshAdapter(core));

    const chunks = [];
    for await (const chunk of ctx.llm.stream({
      provider: "cursor-acp",
      model: "auto",
      tools: [
        { name: "shell", description: "run", parameters: { type: "object" } },
        { name: "hidden", description: "later", parameters: { type: "object" }, deferLoading: true },
      ],
      messages: [
        { role: "developer", content: [{ type: "tool-addition", toolName: "read" }] },
        {
          role: "user",
          content: [
            { type: "text", text: "Look" },
            {
              type: "image",
              attachment: {
                attachmentId: "sha256:0123456789abcdef",
                mediaType: "image/png",
                width: 1,
                height: 1,
              },
            },
            {
              type: "file",
              attachment: {
                attachmentId: "sha256:abcdef0123456789",
                name: "notes.txt",
                bytes: 4,
                mediaType: "text/plain",
              },
            },
          ],
        },
      ] as never,
    })) {
      chunks.push(chunk);
    }

    const prompt = String(bodies[0]?.messages.map((message) => message.content).join("\n"));
    expect(prompt).toContain("Look");
    expect(prompt).toContain("image omitted because this model accepts text only; attachment sha256:01234567");
    expect(prompt).toContain('File "notes.txt" (4 bytes, sha256:abcdef01)');
    expect(prompt).not.toContain("Tool added");
    expect(bodies[0]?.tools?.map((tool) => tool.function.name)).toEqual(["shell", "hidden"]);
    expect(chunks.at(-1)).toMatchObject({ type: "finish", reason: { kind: "stop" } });

    const assembled = new BlockAssembler();
    for (const chunk of chunks) assembled.push(chunk);
    expect(assembled.blocks()).toEqual([{ type: "text", text: "Done" }]);
    expect(ctx.llm.listProviders()).toEqual([{ id: "cursor-acp", name: "Cursor" }]);
    expect(name).toBe("llm-cursor");
    expect(SETTINGS_NS).toBe("llm-cursor");
  });

  test("apply registers cursor-acp and answers model discovery from that catalog", async () => {
    const previousExecutable = process.env.CURSOR_AGENT_EXECUTABLE;
    const previousKey = process.env.CURSOR_API_KEY;
    process.env.CURSOR_AGENT_EXECUTABLE = "/no/such/cursor-agent";
    delete process.env.CURSOR_API_KEY;
    const ctx = new Context();
    try {
      await ctx.plugin(LlmRuntime);
      await ctx.plugin({ name, inject: ["llm"], apply, Config }, {
        apiKey: "test-key",
        baseURL: "http://127.0.0.1:9/v1",
      });
      expect(ctx.llm.listProviders()).toEqual([{ id: "cursor-acp", name: "Cursor" }]);
      expect(ctx.llm.listConfigurableProviders()[0]).toMatchObject({
        provider: "cursor-acp",
        displayName: "Cursor",
        settingsNs: SETTINGS_NS,
      });
      const discovered = await ctx.llm.discoverModels(SETTINGS_NS, { provider: "cursor-acp" });
      expect(discovered.some((model) => model.id === "auto")).toBe(true);
      expect(discovered.every((model) => model.inputModalities?.includes("text"))).toBe(true);
    } finally {
      setStoredApiKey(undefined);
      if (previousExecutable === undefined) delete process.env.CURSOR_AGENT_EXECUTABLE;
      else process.env.CURSOR_AGENT_EXECUTABLE = previousExecutable;
      if (previousKey === undefined) delete process.env.CURSOR_API_KEY;
      else process.env.CURSOR_API_KEY = previousKey;
    }
  });

  test("an aborted signal finishes without calling the proxy", async () => {
    const controller = new AbortController();
    controller.abort();
    let calls = 0;
    const adapter = new CursorHarnessAdapter({
      baseURL: "http://127.0.0.1:9/v1",
      fetch: async () => {
        calls += 1;
        return new Response("nope");
      },
    });
    const chunks = await collect(adapter.stream({
      provider: "cursor-acp",
      model: "auto",
      messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
      signal: controller.signal,
    }, {}));
    expect(calls).toBe(0);
    expect(chunks.map((chunk) => chunk.type)).toEqual(["usage", "finish"]);
    expect(chunks[1]).toMatchObject({ type: "finish", reason: { kind: "aborted" } });
  });

  test("an abort mid-stream closes the open block once", async () => {
    const controller = new AbortController();
    async function* payloads() {
      yield JSON.stringify({ choices: [{ delta: { content: "Hi" }, finish_reason: null }] });
      controller.abort();
      yield JSON.stringify({ choices: [{ delta: { content: " there" }, finish_reason: "stop" }] });
    }
    const chunks = await collect(chunksFromSsePayloads(payloads(), controller.signal));
    const finishes = chunks.filter((chunk) => chunk.type === "finish");
    expect(finishes).toHaveLength(1);
    expect(finishes[0]).toMatchObject({ reason: { kind: "aborted" } });
    expect(chunks.some((chunk) => chunk.type === "block-end" && (chunk.block as { text?: string }).text === "Hi")).toBe(true);
    expect(chunks.some((chunk) => chunk.type === "text-delta" && chunk.text === " there")).toBe(false);
  });

  test("harness errors keep their code and image offload count", () => {
    const error = asLlmError(new CursorHarnessError(
      "offload",
      "IMAGE_OFFLOAD_REQUIRED",
      { offloadImages: 2, status: 413 },
    ));
    expect(error).toBeInstanceOf(LlmError);
    expect(error.code).toBe("IMAGE_OFFLOAD_REQUIRED");
    expect(error.failure.offloadImages).toBe(2);
    expect(error.failure.status).toBe(413);
  });
});

async function expectReject(
  source: AsyncIterable<ChunkLike>,
  code: string,
): Promise<CursorHarnessError> {
  try {
    await collect(source);
  } catch (error) {
    expect(error).toBeInstanceOf(CursorHarnessError);
    expect((error as CursorHarnessError).code).toBe(code);
    return error as CursorHarnessError;
  }
  throw new Error(`expected ${code}`);
}
