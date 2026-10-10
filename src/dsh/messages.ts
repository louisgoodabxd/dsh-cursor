/**
 * Project DeepSeek Harness messages onto the OpenAI chat body the local
 * Cursor proxy already understands (roles, tool_calls, tool results).
 */

export interface ContentBlockLike {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  arguments?: string;
  toolName?: string;
  offloaded?: boolean;
  attachment?: {
    name?: string;
    attachmentId?: string;
    bytes?: number;
  };
}

export interface MessageLike {
  role: string;
  content?: readonly ContentBlockLike[];
  toolCallId?: string;
  isError?: boolean;
}

export interface ToolSchemaLike {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  deferLoading?: boolean;
}

export interface OpenAiChatBody {
  model: string;
  cursorModel: string;
  messages: Array<Record<string, unknown>>;
  tools?: Array<Record<string, unknown>>;
  stream: true;
  temperature?: number;
  max_tokens?: number;
}

const CURSOR_PROVIDER_PREFIX = "cursor-acp/";

/** Model id the Cursor runtime accepts, without the OpenCode provider prefix. */
export function cursorModelId(model: string | undefined): string {
  const trimmed = typeof model === "string" ? model.trim() : "";
  if (!trimmed) return "auto";
  if (trimmed.startsWith(CURSOR_PROVIDER_PREFIX)) {
    const stripped = trimmed.slice(CURSOR_PROVIDER_PREFIX.length).trim();
    return stripped.length > 0 ? stripped : "auto";
  }
  return trimmed;
}

export function countLiveImages(messages: readonly MessageLike[]): number {
  let count = 0;
  for (const message of messages) {
    for (const block of message.content ?? []) {
      if (block?.type === "image" && block.offloaded !== true) count += 1;
    }
  }
  return count;
}

function imagePlaceholder(block: ContentBlockLike): string {
  const attachment = block.attachment;
  const label = attachment?.name || attachment?.attachmentId || "untitled";
  return `[image omitted to fit request image limits; ${label}.]`;
}

function filePlaceholder(block: ContentBlockLike): string {
  const attachment = block.attachment;
  const name = attachment?.name || attachment?.attachmentId || "file";
  const bytes = typeof attachment?.bytes === "number" ? `, ${attachment.bytes} bytes` : "";
  return `[file ${name}${bytes}]`;
}

function renderTextual(block: ContentBlockLike): string | undefined {
  switch (block.type) {
    case "text":
    case "reasoning":
      return typeof block.text === "string" && block.text.length > 0 ? block.text : undefined;
    case "image":
      return block.offloaded === true ? imagePlaceholder(block) : undefined;
    case "file":
      return filePlaceholder(block);
    case "tool-addition":
      return block.toolName ? `Tool added: ${block.toolName}` : undefined;
    case "tool-removal":
      return block.toolName ? `Tool removed: ${block.toolName}` : undefined;
    case "tool-call":
      return undefined;
    default:
      throw new Error(`UNSUPPORTED_BLOCK:${block.type}`);
  }
}

function joinText(blocks: readonly ContentBlockLike[] | undefined, includeReasoning: boolean): string {
  const parts: string[] = [];
  for (const block of blocks ?? []) {
    if (!block || block.type === "tool-call") continue;
    if (block.type === "reasoning" && !includeReasoning) continue;
    const text = renderTextual(block);
    if (text) parts.push(text);
  }
  return parts.join("\n");
}

function toolCallsOf(blocks: readonly ContentBlockLike[] | undefined): Array<Record<string, unknown>> {
  const calls: Array<Record<string, unknown>> = [];
  for (const block of blocks ?? []) {
    if (!block || block.type !== "tool-call") continue;
    const name = typeof block.name === "string" && block.name.length > 0 ? block.name : "tool";
    const id = typeof block.id === "string" && block.id.length > 0 ? block.id : `call_${calls.length}`;
    const args = typeof block.arguments === "string" && block.arguments.length > 0 ? block.arguments : "{}";
    calls.push({
      id,
      type: "function",
      function: { name, arguments: args },
    });
  }
  return calls;
}

/**
 * Build the chat-completions body for one harness request.
 * Throws `UNSUPPORTED_BLOCK:<type>` for content this projection cannot represent.
 * Live images are counted by the caller and rejected before this runs.
 */
export function buildChatBody(input: {
  model: string;
  messages: readonly MessageLike[];
  system?: string;
  tools?: readonly ToolSchemaLike[];
  temperature?: number;
  maxTokens?: number;
}): OpenAiChatBody {
  const modelId = cursorModelId(input.model);
  const messages: Array<Record<string, unknown>> = [];

  if (typeof input.system === "string" && input.system.length > 0) {
    messages.push({ role: "system", content: input.system });
  }
  if (typeof input.maxTokens === "number" && Number.isFinite(input.maxTokens) && input.maxTokens > 0) {
    messages.push({
      role: "system",
      content: `Keep the visible reply within ${Math.floor(input.maxTokens)} tokens.`,
    });
  }

  for (const message of input.messages) {
    const role = message.role;
    if (role === "tool") {
      const text = joinText(message.content, false);
      const body = message.isError ? `ERROR: ${text || "tool failed"}` : (text || "(no output)");
      messages.push({
        role: "tool",
        tool_call_id: message.toolCallId || "unknown",
        content: body,
      });
      continue;
    }

    if (role === "assistant") {
      const calls = toolCallsOf(message.content);
      const text = joinText(message.content, false);
      if (calls.length === 0 && text.length === 0) continue;
      const entry: Record<string, unknown> = {
        role: "assistant",
        content: text,
      };
      if (calls.length > 0) entry.tool_calls = calls;
      messages.push(entry);
      continue;
    }

    const text = joinText(message.content, true);
    if (text.length === 0 && role !== "system") {
      if (role === "user" || role === "developer") continue;
    }
    if (text.length === 0) continue;
    messages.push({ role: role || "user", content: text });
  }

  const tools = (input.tools ?? [])
    .filter((tool) => tool && tool.deferLoading !== true && typeof tool.name === "string" && tool.name.length > 0)
    .map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description ?? "",
        parameters: tool.parameters ?? { type: "object", properties: {} },
      },
    }));

  const body: OpenAiChatBody = {
    model: `${CURSOR_PROVIDER_PREFIX}${modelId}`,
    cursorModel: modelId,
    messages,
    stream: true,
  };
  if (tools.length > 0) body.tools = tools;
  if (typeof input.temperature === "number" && Number.isFinite(input.temperature)) {
    body.temperature = input.temperature;
  }
  if (typeof input.maxTokens === "number" && Number.isFinite(input.maxTokens) && input.maxTokens > 0) {
    body.max_tokens = Math.floor(input.maxTokens);
  }
  return body;
}
