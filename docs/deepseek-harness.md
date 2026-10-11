# DeepSeek Harness

`open-cursor` registers a DeepSeek Harness provider, `cursor-acp`, that
streams Cursor subscription models through the same local proxy the OpenCode
plugin uses. Harness keeps its own tool loop: tool calls that match the
tools Harness advertised are returned to Harness, and everything else stays
inside Cursor.

The package root is still the OpenCode plugin. Harness loads the `dsh`
subpath from the bundle patch, so one install does not replace the other.

## Install

Node.js ≥ 20, [Bun](https://bun.sh) (git installs run `prepare`, which bundles
with Bun), a Cursor API key from [cursor.com/settings](https://cursor.com/settings),
and DeepSeek Harness 0.2 (the `LlmAdapter` contract this plugin implements).

```bash
export CURSOR_API_KEY=...   # the key the proxy's SDK backend reads
dsh plugin --profile web add github:louisgoodabxd/dsh-cursor
```

pnpm will refuse the git dependency's `prepare` build until the profile
allowlists it. Add the package key pnpm prints to the profile's
`pnpm-workspace.yaml`, then run the add again:

```yaml
allowBuilds:
  '@rama_nigg/open-cursor': true
```

From a checkout of this repo:

```bash
dsh plugin --profile web add ./
```

Restart Harness after the bundle is recorded. `dsh --profile web --dump-config`
should show an `llm-cursor` row under the `@rama_nigg/open-cursor` layer.

## Choose a model

The adapter advertises whatever Cursor model discovery returns (static
fallback when no key is available yet). In the model picker, choose provider
**Cursor** and a model id such as `auto` or `composer-2`.

A profile can pin the agent directly:

```yaml
- id: agent-loop
  name: '@deepseek-ai/dsh-agent-loop'
  config:
    agents:
      - id: main
        provider: cursor-acp
        model: auto
```

That override replaces the whole `agent-loop` row, so copy the rest of the
row from your current config before saving it.

## Configuration

The bundle ships no secrets. Optional keys live on the `llm-cursor` plugin
row (or `llm-cursor:` in Harness settings, when that section is merged):

| Key | Default | Role |
|---|---|---|
| `apiKey` | `CURSOR_API_KEY` | Sent as `Authorization` and stored for the proxy's SDK backend |
| `workspaceDirectory` | process working directory | Directory Cursor runs in (`x-opencode-directory`) |
| `providers` | `['cursor-acp']` | Provider routes this adapter owns |
| `contextWindow` | `200000` | Context size reported to Harness |
| `baseURL` | local proxy | OpenAI-compatible root, including `/v1`, when a proxy is already running |

`stop` sequences are rejected with `UNSUPPORTED_OPTION`. Cursor has no stop
field. `maxTokens` is forwarded on the request and also written into the
prompt as a length instruction, because the agent backend does not enforce an
output cap. `temperature` is forwarded on the JSON body only.

The model list declares input modality `text`. Harness rewrites image blocks
to its text-only placeholder before `stream()` runs, and rewrites file blocks
to handle text the same way. An image block that still reaches the adapter
fails with `IMAGE_OFFLOAD_REQUIRED` and `offloadImages` set to the number of
live images, so the offload executor can drop them and retry. Images already
marked offloaded are sent as the harness offload placeholder.

HTTP 408 and 504 are `TIMEOUT`, other 5xx responses are `SERVER`, and a
failed connection is `TRANSPORT`. Those three are in the default retry set.
401 and 403 stay `AUTH`. A key that contains characters an HTTP header cannot
carry is refused as `INVALID_CREDENTIAL` when the plugin loads.

## Tools

Leave `CURSOR_ACP_TOOL_LOOP_MODE` at its default, `opencode`. Harness tool
names that match a Cursor call (or a known alias such as `bash` → `shell`)
are returned as harness tool calls. Other calls run inside Cursor. This
route does not advertise mid-conversation tool updates, so Harness sends the
current tool list on every request, including tools that were marked for
deferred loading.

The workspace directory is the one configured above. Harness does not send a
per-session project path on `GenerateOptions`.
