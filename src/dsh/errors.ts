/**
 * Failures the DeepSeek Harness adapter raises before they are wrapped as
 * `LlmError`. Codes match the harness taxonomy where one exists.
 */
export class CursorHarnessError extends Error {
  readonly code: string;
  readonly status?: number;
  readonly offloadImages?: number;

  constructor(
    message: string,
    code: string,
    options?: { status?: number; offloadImages?: number; cause?: unknown },
  ) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = "CursorHarnessError";
    this.code = code;
    if (options?.status !== undefined) this.status = options.status;
    if (options?.offloadImages !== undefined) this.offloadImages = options.offloadImages;
  }
}

export function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const name = (error as { name?: string }).name;
  return name === "AbortError" || name === "TimeoutError";
}
