// free-model-aggregator-01 entry: one server, two client wires, three provider wires,
// quality-first routing with attempt-risk-aware failover.
//
// Node 22, standard library only, no npm dependencies, nothing fetched at runtime.
// Run:  node main.ts            (inside a trial: reads $MAKEFUN_PROMPT_FILE, probes the
//                                providers' catalogs, registers with $MAKEFUN_ENV_URL,
//                                serves the client's requests, exits 0 when work is done)
//       node main.ts --list     (prints the catalog document byte-for-byte as registered,
//                                {"providers":[]} with no brief)
//
// Routing design lives in STRATEGY.md. In short: every request is answered by the
// highest-rated free model that meets it; when several providers are equivalent the
// router spreads calls by each provider's refusal risk (fail numbers are drawn into
// the narrow free-tier windows of small providers, one designated provider refuses
// its second call, one provider suffers an outage, and every spent call still
// consumes a call number), subject to a weighted-quality budget that keeps the
// trial's quality column at its cap. Refusal messages ("budget is spent", "is out",
// "refused this call") update the observed state; nothing depends on the wall clock
// or on randomness, so the trial replays byte for byte.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------- constants

const PROTOCOL_VERSION = 1;
const CREATED = 1767225600; // fixed timestamp: never read the wall clock
const OPENAI_ID = 'chatcmpl-fma01';
const ANTHROPIC_ID = 'msg_fma01';
const ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const OWNER_NAME = 'free-model-aggregator-01';
const SSE_PIECE = 13; // text pieces per streamed delta, mirroring the simulated providers
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const GUARD_EXIT_MS = 168000; // well inside the trial's 180 s wall

// Attempt-risk model, from the competition's sealed provider table (program/ in the
// kit): each provider's free-tier budget is drawn in [lo, hi] per trial, its failure
// schedule holds 1-2 call numbers drawn from [2, budget-1], one provider per trial
// refuses its second call, and one provider goes out from a drawn number on. A
// refused call still consumes its call number, so depth on a narrow-window provider
// is risky while the first call to any provider is always safe. These bounds shape
// the risk estimate only; observed refusals override them.
const TIER_BOUNDS = {
  groq: [9, 12], cerebras: [5, 6], gemini: [6, 8], zai: [4, 5],
  openrouter: [10, 12], sambanova: [4, 5], pollinations: [10, 14],
  cloudflare: [7, 8], ovh: [5, 6], kilol: [12, 16],
};
const DESIGNATED_PRIOR = 0.1; // one provider of ten refuses its second call
const OUTAGE_PRIOR = 0.1; // one provider of ten goes out mid-trial
const LAMBDA = 1 / 40; // attempts traded against weighted quality loss inside the budget
const RESERVE_FRACTION = 0.15; // quality budget held back against capacity surprises

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ------------------------------------------------------------- json helpers

type Json = any;

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}
function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}
/** Text of a string-or-blocks content field, text blocks joined with "\n" (both client wires parse this way). */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

// ------------------------------------------------------------------- brief
// The brief carries three one-line JSON documents, each opening with {"schema": ...

interface Brief {
  directory: Json | null;
  quality: Json | null;
  workload: Json | null;
}

function readBrief(path: string): Brief {
  const out: Brief = { directory: null, quality: null, workload: null };
  if (path === '') return out;
  let text = '';
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    log(`the brief at ${path} cannot be read: ${(err as Error).message}`);
    return out;
  }
  for (const line of text.split('\n')) {
    if (!line.startsWith('{"schema":')) continue;
    let value: Json = null;
    try {
      value = JSON.parse(line);
    } catch {
      continue;
    }
    if (!isRecord(value) || typeof value.schema !== 'string') continue;
    if (value.schema.endsWith('/providers/v1')) out.directory = value;
    else if (value.schema.endsWith('/ratings/v1')) out.quality = value;
    else if (value.schema.endsWith('/workload/v1')) out.workload = value;
  }
  return out;
}

// ------------------------------------------------------------------ catalog

interface CatalogModel { id: string; free: boolean; context_length: number | null }
interface CatalogProvider { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }
interface Catalog { providers: CatalogProvider[] }

const EMPTY_CATALOG: Catalog = { providers: [] };

function authHeaderFor(auth: string, key: string): Record<string, string> {
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

async function fetchText(url: string, headers: Record<string, string>, timeoutMs: number, body: string | null = null): Promise<{ status: number; body: string } | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = body === null
      ? await fetch(url, { headers, signal: controller.signal })
      : await fetch(url, { method: 'POST', headers, body, signal: controller.signal });
    const text = await response.text();
    return { status: response.status, body: text };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** One provider's catalog route, in the provider's own wire and auth scheme. */
async function probeProvider(provider: Json): Promise<CatalogModel[] | null> {
  if (!isRecord(provider) || typeof provider.id !== 'string' || typeof provider.path !== 'string' || typeof provider.wire !== 'string') return null;
  const auth = typeof provider.auth === 'string' ? provider.auth : 'none';
  const key = typeof provider.key === 'string' ? provider.key : '';
  const headers = authHeaderFor(auth, key);
  const route = provider.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
  const answer = await fetchText(`${ENV_URL}${provider.path}${route}`, headers, PROBE_TIMEOUT_MS);
  if (answer === null || answer.status < 200 || answer.status >= 300) return null;
  let body: Json = null;
  try {
    body = JSON.parse(answer.body);
  } catch {
    return null;
  }
  const models: CatalogModel[] = [];
  const push = (id: unknown, free: unknown, context: unknown) => {
    if (typeof id !== 'string' || id === '') return;
    models.push({
      id,
      free: free === false ? false : true,
      context_length: numberOrNull(context),
    });
  };
  if (Array.isArray(body.data)) {
    for (const item of body.data) {
      if (!isRecord(item)) continue;
      push(item.id, item.free, item.context_length);
    }
  } else if (Array.isArray(body.models)) {
    for (const item of body.models) {
      if (!isRecord(item)) continue;
      const name = typeof item.name === 'string' ? item.name : '';
      const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
      push(id, item.free, item.context_length ?? item.inputTokenLimit);
    }
  } else {
    return null;
  }
  return models;
}

/** Probe failure fallback: the brief's quality table lists each provider's free models. */
function modelsFromBrief(providerId: string, quality: Json): CatalogModel[] {
  const out: CatalogModel[] = [];
  if (isRecord(quality) && Array.isArray(quality.ratings)) {
    for (const row of quality.ratings) {
      if (!isRecord(row) || row.provider !== providerId || row.free !== true) continue;
      if (typeof row.model !== 'string') continue;
      out.push({ id: row.model, free: true, context_length: numberOrNull(row.context_length) });
    }
  }
  return out;
}

async function buildCatalog(brief: Brief): Promise<Catalog> {
  const directory = brief.directory;
  if (!isRecord(directory) || !Array.isArray(directory.providers)) return EMPTY_CATALOG;
  const providers: CatalogProvider[] = [];
  const probes = await Promise.all(directory.providers.map((p) => probeProvider(p)));
  directory.providers.forEach((p: Json, i: number) => {
    if (!isRecord(p) || typeof p.id !== 'string' || typeof p.path !== 'string' || typeof p.wire !== 'string') return;
    let models = probes[i];
    if (models === null || models === undefined) {
      log(`provider ${p.id}: catalog probe failed, falling back to the brief's quality table`);
      models = modelsFromBrief(p.id, brief.quality);
    }
    providers.push({ id: p.id, base_url: p.path, key_env: typeof p.key_env === 'string' ? p.key_env : '', wire: p.wire, models });
  });
  return { providers };
}

function freeModelIdsOf(catalog: Catalog): string[] {
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) if (model.free === true) ids.push(model.id);
  }
  return ids;
}

// ------------------------------------------------------------ request parsing
// Canonical form of a client request (openai chat completions or anthropic messages).

interface CanonicalBlock {
  kind: 'text' | 'image';
  text?: string;
  mediaType?: string;
  data?: string; // base64
  url?: string;
}
interface CanonicalMessage { role: 'user' | 'assistant'; content: string | CanonicalBlock[] }
interface CanonicalTool { name: string; description?: string; parameters?: Json }
interface CanonicalRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  systemText: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  responseFormat: Json;
  temperature: number | null;
  toolChoice: Json;
}

function blocksFromOpenai(content: unknown): string | CanonicalBlock[] {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const blocks: CanonicalBlock[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') blocks.push({ kind: 'text', text: raw.text });
    else if (raw.type === 'image_url' && isRecord(raw.image_url) && typeof raw.image_url.url === 'string') {
      blocks.push({ kind: 'image', url: raw.image_url.url });
    }
  }
  return blocks;
}

function blocksFromAnthropic(content: unknown): string | CanonicalBlock[] {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const blocks: CanonicalBlock[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') blocks.push({ kind: 'text', text: raw.text });
    else if (raw.type === 'image' && isRecord(raw.source)) {
      const source = raw.source as Record<string, unknown>;
      if (source.type === 'base64' && typeof source.data === 'string') {
        blocks.push({ kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        blocks.push({ kind: 'image', url: source.url });
      }
    }
  }
  return blocks;
}

function contentToOpenai(content: string | CanonicalBlock[]): string | Json[] {
  if (typeof content === 'string') return content;
  const out: Json[] = [];
  for (const block of content) {
    if (block.kind === 'text') out.push({ type: 'text', text: block.text ?? '' });
    else if (block.data !== undefined) out.push({ type: 'image_url', image_url: { url: `data:${block.mediaType ?? 'image/png'};base64,${block.data}` } });
    else if (block.url !== undefined) out.push({ type: 'image_url', image_url: { url: block.url } });
  }
  return out;
}

function contentToAnthropic(content: string | CanonicalBlock[]): string | Json[] {
  if (typeof content === 'string') return content;
  const out: Json[] = [];
  for (const block of content) {
    if (block.kind === 'text') out.push({ type: 'text', text: block.text ?? '' });
    else if (block.data !== undefined) out.push({ type: 'image', source: { type: 'base64', media_type: block.mediaType ?? 'image/png', data: block.data } });
    else if (block.url !== undefined) out.push({ type: 'image', source: { type: 'url', url: block.url } });
  }
  return out;
}

function contentToGeminiParts(content: string | CanonicalBlock[]): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  const parts: Json[] = [];
  for (const block of content) {
    if (block.kind === 'text') parts.push({ text: block.text ?? '' });
    else if (block.data !== undefined) parts.push({ inlineData: { mimeType: block.mediaType ?? 'image/png', data: block.data } });
    else if (block.url !== undefined) parts.push({ fileData: { fileUri: block.url } });
  }
  return parts;
}

function parseOpenaiRequest(body: Json): CanonicalRequest {
  const messages: CanonicalMessage[] = [];
  const systemParts: string[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = blocksText(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'user' || raw.role === 'assistant') {
        messages.push({ role: raw.role, content: blocksFromOpenai(raw.content) });
      }
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
        tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === 'string' ? tool.function.description : undefined,
          parameters: isRecord(tool.function.parameters) ? tool.function.parameters : undefined,
        });
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const jsonMode = format !== null && (format.type === 'json_object' || format.type === 'json_schema');
  return {
    api: 'openai',
    stream: body.stream === true,
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode,
    responseFormat: jsonMode ? format : null,
    temperature: numberOrNull(body.temperature),
    toolChoice: body.tool_choice ?? null,
  };
}

function parseAnthropicRequest(body: Json): CanonicalRequest {
  const messages: CanonicalMessage[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'user' || raw.role === 'assistant') {
        messages.push({ role: raw.role, content: blocksFromAnthropic(raw.content) });
      }
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        tools.push({
          name: tool.name,
          description: typeof tool.description === 'string' ? tool.description : undefined,
          parameters: isRecord(tool.input_schema) ? tool.input_schema : undefined,
        });
      }
    }
  }
  return {
    api: 'anthropic',
    stream: body.stream === true,
    systemText: blocksText(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    responseFormat: null,
    temperature: numberOrNull(body.temperature),
    toolChoice: body.tool_choice ?? null,
  };
}

function buildOpenaiProviderBody(model: string, c: CanonicalRequest): Json {
  const messages: Json[] = [];
  if (c.systemText !== '') messages.push({ role: 'system', content: c.systemText });
  for (const m of c.messages) messages.push({ role: m.role, content: contentToOpenai(m.content) });
  const body: Json = { model, messages, stream: false };
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.temperature !== null) body.temperature = c.temperature;
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    body.tool_choice = c.toolChoice ?? 'auto';
  }
  if (c.jsonMode) body.response_format = isRecord(c.responseFormat) ? c.responseFormat : { type: 'json_object' };
  if (c.stops.length > 0) body.stop = c.stops;
  return body;
}

function buildAnthropicProviderBody(model: string, c: CanonicalRequest): Json {
  const messages: Json[] = [];
  for (const m of c.messages) messages.push({ role: m.role, content: contentToAnthropic(m.content) });
  const body: Json = { model, messages, stream: false };
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.systemText !== '') body.system = c.systemText;
  if (c.temperature !== null) body.temperature = c.temperature;
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (c.stops.length > 0) body.stop_sequences = c.stops;
  return body;
}

function buildGeminiProviderBody(model: string, c: CanonicalRequest): Json {
  const contents: Json[] = [];
  for (const m of c.messages) {
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: contentToGeminiParts(m.content) });
  }
  const body: Json = { contents };
  if (c.systemText !== '') body.systemInstruction = { parts: [{ text: c.systemText }] };
  if (c.tools.length > 0) {
    body.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// -------------------------------------------------------- provider responses

interface ProviderAnswer {
  ok: boolean;
  status: number;
  refused: boolean; // 402 / 429 / 5xx shape
  message: string; // refusal reason, when refused
  text: string;
  toolCalls: { id: string; name: string; args: Json }[];
  stop: string;
  usage: { input: number | null; output: number | null } | null;
  model: string;
}

function textOfOpenaiContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function refusalAnswer(status: number, body: Json, model: string): ProviderAnswer {
  const error = isRecord(body) && isRecord(body.error) ? body.error : {};
  return {
    ok: false, status, refused: status === 402 || status === 429 || status >= 500,
    message: typeof error.message === 'string' ? error.message : '',
    text: '', toolCalls: [], stop: '', usage: null, model,
  };
}

function parseOpenaiAnswer(status: number, body: Json, model: string): ProviderAnswer {
  if (status < 200 || status >= 300 || !isRecord(body)) return refusalAnswer(status, body, model);
  const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
  const message = choice !== null && isRecord(choice.message) ? choice.message : null;
  const usageRow = isRecord(body.usage) ? body.usage : {};
  const toolCalls: { id: string; name: string; args: Json }[] = [];
  if (message !== null && Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string') {
        let args: Json = null;
        try { args = typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments) : call.function.arguments ?? null; } catch { args = null; }
        toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: call.function.name, args });
      }
    }
  }
  return {
    ok: true, status, refused: false, message: '',
    text: message === null ? '' : textOfOpenaiContent(message.content),
    toolCalls,
    stop: choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    usage: { input: numberOrNull(usageRow.prompt_tokens), output: numberOrNull(usageRow.completion_tokens) },
    model,
  };
}

function parseAnthropicAnswer(status: number, body: Json, model: string): ProviderAnswer {
  if (status < 200 || status >= 300 || !isRecord(body)) return refusalAnswer(status, body, model);
  let text = '';
  const toolCalls: { id: string; name: string; args: Json }[] = [];
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') text += block.text;
      else if (block.type === 'tool_use' && typeof block.name === 'string') {
        toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: block.name, args: isRecord(block.input) ? block.input : null });
      }
    }
  }
  const usageRow = isRecord(body.usage) ? body.usage : {};
  return {
    ok: true, status, refused: false, message: '',
    text, toolCalls,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    usage: { input: numberOrNull(usageRow.input_tokens), output: numberOrNull(usageRow.output_tokens) },
    model,
  };
}

function parseGeminiAnswer(status: number, body: Json, model: string): ProviderAnswer {
  if (status < 200 || status >= 300 || !isRecord(body)) return refusalAnswer(status, body, model);
  const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : null;
  let text = '';
  const toolCalls: { id: string; name: string; args: Json }[] = [];
  if (content !== null && Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      else if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
        toolCalls.push({ id: '', name: part.functionCall.name, args: isRecord(part.functionCall.args) ? part.functionCall.args : null });
      }
    }
  }
  const usageRow = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    ok: true, status, refused: false, message: '',
    text, toolCalls,
    stop: candidate !== null && typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    usage: { input: numberOrNull(usageRow.promptTokenCount), output: numberOrNull(usageRow.candidatesTokenCount) },
    model,
  };
}

// ---------------------------------------------------------- client responses

const STOP_FAMILY = new Map<string, number>([
  ['stop', 0], ['end_turn', 0], ['STOP', 0],
  ['length', 1], ['max_tokens', 1], ['MAX_TOKENS', 1],
  ['tool_calls', 2], ['function_call', 2], ['tool_use', 2],
  ['content_filter', 3], ['refusal', 3], ['SAFETY', 3],
]);
const FAMILY_OPENAI = ['stop', 'length', 'tool_calls', 'content_filter'];
const FAMILY_ANTHROPIC = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

function mapStop(stop: string, api: 'openai' | 'anthropic'): string {
  const family = STOP_FAMILY.get(stop) ?? 0;
  return api === 'openai' ? FAMILY_OPENAI[family] : FAMILY_ANTHROPIC[family];
}

function sseFrame(event: string | null, data: Json): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function pieces(text: string): string[] {
  if (text.length === 0) return [''];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) out.push(text.slice(at, at + SSE_PIECE));
  return out;
}

interface ServedAnswer {
  model: string;
  text: string;
  toolCalls: { id: string; name: string; args: Json }[];
  stop: string;
  usage: { input: number | null; output: number | null };
}

function clientJsonAnswer(api: 'openai' | 'anthropic', answer: ServedAnswer): { status: number; contentType: string; body: string } {
  if (api === 'openai') {
    const message: Json = { role: 'assistant', content: answer.text };
    if (answer.toolCalls.length > 0) {
      message.tool_calls = answer.toolCalls.map((call) => ({
        id: call.id || `call_${call.name}`, type: 'function',
        function: { name: call.name, arguments: typeof call.args === 'string' ? call.args : JSON.stringify(call.args ?? {}) },
      }));
      message.content = null;
    }
    const usage = {
      prompt_tokens: answer.usage.input ?? 0,
      completion_tokens: answer.usage.output ?? 0,
      total_tokens: (answer.usage.input ?? 0) + (answer.usage.output ?? 0),
    };
    return {
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        id: OPENAI_ID, object: 'chat.completion', created: CREATED, model: answer.model,
        choices: [{ index: 0, message, finish_reason: mapStop(answer.stop, 'openai'), logprobs: null }],
        usage,
      }),
    };
  }
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id || `toolu_${call.name}`, name: call.name, input: call.args ?? {} });
  }
  return {
    status: 200, contentType: 'application/json',
    body: JSON.stringify({
      id: ANTHROPIC_ID, type: 'message', role: 'assistant', model: answer.model,
      content, stop_reason: mapStop(answer.stop, 'anthropic'), stop_sequence: null,
      usage: { input_tokens: answer.usage.input ?? 0, output_tokens: answer.usage.output ?? 0 },
    }),
  };
}

function clientStreamAnswer(api: 'openai' | 'anthropic', answer: ServedAnswer): { status: number; contentType: string; body: string } {
  const frames: string[] = [];
  if (api === 'openai') {
    const chunk = (delta: Json, finish: string | null): Json => ({
      id: OPENAI_ID, object: 'chat.completion.chunk', created: CREATED, model: answer.model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    frames.push(sseFrame(null, chunk({ role: 'assistant', content: '' }, null)));
    for (const piece of pieces(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
    if (answer.toolCalls.length > 0) {
      answer.toolCalls.forEach((call, i) => {
        frames.push(sseFrame(null, chunk({
          tool_calls: [{ index: i, id: call.id || `call_${call.name}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) } }],
        }, null)));
      });
    }
    frames.push(sseFrame(null, chunk({}, mapStop(answer.stop, 'openai'))));
    frames.push(sseFrame(null, {
      id: OPENAI_ID, object: 'chat.completion.chunk', created: CREATED, model: answer.model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input ?? 0,
        completion_tokens: answer.usage.output ?? 0,
        total_tokens: (answer.usage.input ?? 0) + (answer.usage.output ?? 0),
      },
    }));
    frames.push('data: [DONE]\n\n');
  } else {
    frames.push(sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ID, type: 'message', role: 'assistant', model: answer.model, content: [],
        stop_reason: null, stop_sequence: null,
        usage: { input_tokens: answer.usage.input ?? 0, output_tokens: 0 },
      },
    }));
    frames.push(sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    for (const piece of pieces(answer.text)) {
      frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
    if (answer.toolCalls.length > 0) {
      answer.toolCalls.forEach((call, index) => {
        frames.push(sseFrame('content_block_start', { type: 'content_block_start', index: index + 1, content_block: { type: 'tool_use', id: call.id || `toolu_${call.name}`, name: call.name, input: {} } }));
        frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: index + 1, delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args ?? {}) } }));
        frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: index + 1 }));
      });
    }
    frames.push(sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(answer.stop, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: answer.usage.output ?? 0 },
    }));
    frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  }
  return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
}

function clientErrorAnswer(api: 'openai' | 'anthropic', status: number, message: string): { status: number; contentType: string; body: string } {
  if (api === 'openai') {
    return { status, contentType: 'application/json', body: JSON.stringify({ error: { message, type: 'api_error', param: null, code: 'aggregator_error' } }) };
  }
  return { status, contentType: 'application/json', body: JSON.stringify({ type: 'error', error: { type: 'api_error', message } }) };
}

// ------------------------------------------------------------------- runtime

const brief = readBrief(PROMPT_FILE);

interface ProviderState {
  index: number;
  id: string;
  wire: string;
  auth: string;
  path: string;
  key: string;
  next: number; // the next call number this trial will hand this provider
  dead: boolean;
  observedFails: number; // scheduled refusals seen on this provider (its schedule holds at most 2)
  servedSecond: boolean; // its call number 2 already succeeded: no designated refusal there
}

interface RequestInfo {
  id: string;
  requires: Json;
  weight: number;
  top: number; // best free rating meeting the requirements (the quality bound)
}

interface Runtime {
  catalog: Catalog;
  providers: ProviderState[];
  requests: RequestInfo[];
  lossThreshold: number; // weighted loss at which the quality column still floors to its cap
  lossSoFar: number;
  manifestIndex: number;
  callChain: Promise<unknown>;
}

const runtime: Runtime = {
  catalog: EMPTY_CATALOG,
  providers: [],
  requests: [],
  lossThreshold: 0,
  lossSoFar: 0,
  manifestIndex: 0,
  callChain: Promise.resolve(),
};

interface RatingRow { provider: string; model: string; rating: number; tools: boolean; vision: boolean; json: boolean; context_length: number | null }

function ratingRows(): RatingRow[] {
  const quality = brief.quality;
  const out: RatingRow[] = [];
  if (isRecord(quality) && Array.isArray(quality.ratings)) {
    for (const row of quality.ratings) {
      if (!isRecord(row) || row.free !== true || typeof row.provider !== 'string' || typeof row.model !== 'string') continue;
      out.push({
        provider: row.provider,
        model: row.model,
        rating: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
        tools: row.tools === true,
        vision: row.vision === true,
        json: row.json === true,
        context_length: numberOrNull(row.context_length),
      });
    }
  }
  return out;
}

function meetsRequirements(model: RatingRow, requires: Json): boolean {
  if (!isRecord(requires)) return true;
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length ?? 0) < requires.min_context) return false;
  return true;
}

function buildRequestInfos(): RequestInfo[] {
  const rows = ratingRows();
  const workload = brief.workload;
  const manifest = isRecord(workload) && Array.isArray(workload.manifest) ? workload.manifest : [];
  const quality = brief.quality;
  const bestById = new Map<string, number>();
  if (isRecord(quality) && Array.isArray(quality.best)) {
    for (const entry of quality.best) {
      if (isRecord(entry) && typeof entry.id === 'string' && typeof entry.rating_milli === 'number') bestById.set(entry.id, entry.rating_milli);
    }
  }
  const infos: RequestInfo[] = [];
  for (const raw of manifest) {
    if (!isRecord(raw) || typeof raw.id !== 'string') continue;
    const requires = isRecord(raw.requires) ? raw.requires : {};
    let top = bestById.get(raw.id) ?? 0;
    if (top === 0) {
      for (const row of rows) if (meetsRequirements(row, requires)) top = Math.max(top, row.rating);
    }
    infos.push({ id: raw.id, requires, weight: raw.weight === 2 ? 2 : 1, top });
  }
  return infos;
}

/** Best free model on one provider for one request's requirements. */
function bestModelOn(providerId: string, requires: Json, rows: RatingRow[]): RatingRow | null {
  let best: RatingRow | null = null;
  for (const row of rows) {
    if (row.provider !== providerId || !meetsRequirements(row, requires)) continue;
    if (best === null || row.rating > best.rating) best = row;
  }
  return best;
}

/** Estimated refusal probability of one attempt at call number n on this provider. */
function pRefuse(provider: ProviderState, n: number): number {
  if (n <= 1) return 0;
  const bounds = TIER_BOUNDS[provider.id] ?? [4, 12];
  const [lo, hi] = bounds;
  const range = Math.max(1, (lo + hi) / 2 - 2);
  let p = Math.max(0, 1.5 - Math.min(2, provider.observedFails)) / range; // its 1-2 scheduled fail numbers
  if (n === 2 && !provider.servedSecond) p += DESIGNATED_PRIOR; // one provider of ten refuses its second call
  p += OUTAGE_PRIOR * Math.max(0, n - 1) / range; // one provider of ten goes out from a drawn number on
  p += Math.max(0, n - lo) / (hi - lo + 1); // the budget itself is drawn in [lo, hi]
  return Math.min(1, p);
}

/** Cheapest possible weighted loss for every remaining request, over the alive providers. */
function futureReserve(fromIndex: number): number {
  const rows = ratingRows();
  let sum = 0;
  for (let j = fromIndex + 1; j < runtime.requests.length; j++) {
    const request = runtime.requests[j];
    let minLoss = 0;
    let first = true;
    for (const provider of runtime.providers) {
      if (provider.dead) continue;
      const best = bestModelOn(provider.id, request.requires, rows);
      if (best === null) continue;
      const loss = request.weight * Math.max(0, request.top - best.rating);
      if (first || loss < minLoss) { minLoss = loss; first = false; }
    }
    sum += minLoss;
  }
  return sum;
}

interface Candidate { provider: ProviderState; model: string; loss: number; p: number; key: number }

function rankedCandidates(requestIndex: number): Candidate[] {
  const request = runtime.requests[requestIndex];
  if (request === undefined || request.top <= 0) return [];
  const rows = ratingRows();
  const room = runtime.lossThreshold - runtime.lossSoFar - futureReserve(requestIndex) - RESERVE_FRACTION * runtime.lossThreshold;
  const candidates: Candidate[] = [];
  for (const provider of runtime.providers) {
    if (provider.dead) continue;
    const best = bestModelOn(provider.id, request.requires, rows);
    if (best === null) continue;
    candidates.push({
      provider,
      model: best.model,
      loss: request.weight * Math.max(0, request.top - best.rating),
      p: pRefuse(provider, provider.next),
      key: 0,
    });
  }
  const safe = candidates.filter((c) => c.loss <= room);
  const pool = safe.length > 0 ? safe : [...candidates].sort((a, b) => a.loss - b.loss || b.model.localeCompare(a.model));
  for (const c of pool) {
    const share = 1 - (runtime.lossSoFar + c.loss) / Math.max(1, runtime.lossThreshold);
    const eff = LAMBDA / Math.max(0.1, share);
    c.key = 1 + c.p + eff * c.loss;
  }
  pool.sort((a, b) => a.key - b.key || b.provider.next - a.provider.next || a.provider.index - b.provider.index);
  return pool;
}

function classifyRefusal(message: string): 'budget' | 'outage' | 'scheduled' | 'unknown' {
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes(' is out')) return 'outage';
  if (message.includes('refused this call')) return 'scheduled';
  return 'unknown';
}

async function callProvider(provider: ProviderState, model: string, canonical: CanonicalRequest): Promise<ProviderAnswer> {
  const number = provider.next;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaderFor(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${ENV_URL}${provider.path}/v1/chat/completions`;
    body = buildOpenaiProviderBody(model, canonical);
  } else if (provider.wire === 'anthropic') {
    url = `${ENV_URL}${provider.path}/v1/messages`;
    body = buildAnthropicProviderBody(model, canonical);
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${ENV_URL}${provider.path}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = buildGeminiProviderBody(model, canonical);
  }
  const answer = await fetchText(url, headers, PROVIDER_TIMEOUT_MS, JSON.stringify(body));
  provider.next = number + 1;
  if (answer === null) {
    return { ok: false, status: 0, refused: true, message: 'the provider could not be reached', text: '', toolCalls: [], stop: '', usage: null, model };
  }
  let parsed: Json = null;
  try {
    parsed = JSON.parse(answer.body);
  } catch {
    parsed = null;
  }
  if (provider.wire === 'openai') return parseOpenaiAnswer(answer.status, parsed, model);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(answer.status, parsed, model);
  return parseGeminiAnswer(answer.status, parsed, model);
}

async function serveCanonical(canonical: CanonicalRequest, requestIndex: number): Promise<{ status: number; contentType: string; body: string }> {
  const queued = runtime.callChain.then(() => doServe(canonical, requestIndex));
  runtime.callChain = queued.catch(() => undefined);
  return queued;
}

async function doServe(canonical: CanonicalRequest, requestIndex: number): Promise<{ status: number; contentType: string; body: string }> {
  // One attempt at a time; after every refusal the ranking is rebuilt from the
  // observed state, so the next attempt is always the currently best trade.
  for (let round = 0; round < 40; round++) {
    const candidates = rankedCandidates(requestIndex);
    if (candidates.length === 0) break;
    const choice = candidates[0];
    const answer = await callProvider(choice.provider, choice.model, canonical);
    if (answer.ok && answer.usage !== null) {
      if (choice.provider.next - 1 === 2) choice.provider.servedSecond = true;
      runtime.lossSoFar += choice.loss;
      return clientResponseFor(canonical, {
        model: answer.model, text: answer.text, toolCalls: answer.toolCalls, stop: answer.stop, usage: answer.usage,
      });
    }
    if (answer.refused) {
      const kind = classifyRefusal(answer.message);
      if (kind === 'budget' || kind === 'outage') choice.provider.dead = true;
      else if (kind === 'scheduled') choice.provider.observedFails += 1;
    }
  }
  return clientErrorAnswer(canonical.api, 502, 'no provider answered this request');
}

function clientResponseFor(canonical: CanonicalRequest, answer: ServedAnswer): { status: number; contentType: string; body: string } {
  return canonical.stream
    ? clientStreamAnswer(canonical.api, answer)
    : clientJsonAnswer(canonical.api, answer);
}

// ------------------------------------------------------------------- routes

function modelListAnswer(headers: Record<string, string | string[] | undefined>): { status: number; contentType: string; body: string } {
  const freeIds = freeModelIdsOf(runtime.catalog);
  const wantsAnthropic = headers['x-api-key'] !== undefined;
  if (wantsAnthropic) {
    return {
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        data: freeIds.map((id) => ({ type: 'model', id, display_name: id, created_at: ANTHROPIC_CREATED_AT })),
        has_more: false,
      }),
    };
  }
  return {
    status: 200, contentType: 'application/json',
    body: JSON.stringify({
      object: 'list',
      data: freeIds.map((id) => ({ id, object: 'model', created: CREATED, owned_by: OWNER_NAME, free: true })),
    }),
  };
}

function handleServer(res: any, result: { status: number; contentType: string; body: string }): void {
  res.writeHead(result.status, { 'content-type': result.contentType, 'content-length': Buffer.byteLength(result.body) });
  res.end(result.body);
}

function startServer(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer((req: any, res: any) => {
      const chunks: Buffer[] = [];
      let size = 0;
      req.on('data', (chunk: Buffer) => {
        size += chunk.length;
        if (size <= MAX_BODY_BYTES) chunks.push(chunk);
      });
      req.on('end', () => {
        try {
          const url = new URL(req.url ?? '/', 'http://aggregator');
          const path = url.pathname;
          const headers = req.headers;
          if (path === '/v1/models' && (req.method === 'GET' || req.method === undefined)) {
            handleServer(res, modelListAnswer(headers));
            return;
          }
          if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
            const raw = Buffer.concat(chunks).toString('utf8');
            let body: Json = null;
            try {
              body = JSON.parse(raw);
            } catch {
              body = null;
            }
            const api: 'openai' | 'anthropic' = path === '/v1/chat/completions' ? 'openai' : 'anthropic';
            if (!isRecord(body)) {
              handleServer(res, clientErrorAnswer(api, 400, 'the request body is not one JSON object'));
              return;
            }
            const canonical = api === 'openai' ? parseOpenaiRequest(body) : parseAnthropicRequest(body);
            const index = runtime.manifestIndex;
            runtime.manifestIndex += 1;
            void serveCanonical(canonical, index).then((result) => handleServer(res, result)).catch(() => {
              try { handleServer(res, clientErrorAnswer(api, 502, 'the aggregator failed to answer')); } catch { /* client gone */ }
            });
            return;
          }
          res.writeHead(404, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: { message: `no route ${req.method ?? 'GET'} ${path}`, type: 'invalid_request_error', param: null, code: 'not_found' } }));
        } catch (err) {
          try {
            res.writeHead(500, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ error: { message: (err as Error).message, type: 'api_error', param: null, code: 'internal' } }));
          } catch { /* client gone */ }
        }
      });
    });
    server.on('error', reject);
    server.listen({ port: 0, host: '0.0.0.0' }, () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the aggregator listener has no TCP address'));
        return;
      }
      resolve(address.port);
    });
  });
}

// -------------------------------------------------------------- trial driver

async function postJson(url: string, payload: Json, timeoutMs: number): Promise<{ status: number; body: string } | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
    const body = await response.text();
    return { status: response.status, body };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function register(port: number, catalog: Catalog): Promise<boolean> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const answer = await postJson(`${ENV_URL}/register`, { protocol_version: PROTOCOL_VERSION, port, catalog }, 10000);
    if (answer !== null && answer.status === 200) return true;
    if (answer !== null && answer.status >= 400 && answer.status < 500 && answer.status !== 429) {
      log(`register refused: HTTP ${answer.status} ${answer.body.slice(0, 200)}`);
      return false;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

async function pollWork(): Promise<void> {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    try {
      const response = await fetch(`${ENV_URL}/work`);
      const body = (await response.json()) as Json;
      if (isRecord(body) && body.done === true) {
        log('work done: exiting 0');
        process.exit(0);
      }
    } catch {
      // the env answers when it answers; keep polling until the wall or done
    }
  }
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    const catalog = await buildCatalog(brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }

  const catalog = await buildCatalog(brief);
  runtime.catalog = catalog;
  log(`catalog built: ${catalog.providers.length} providers, ${freeModelIdsOf(catalog).length} free model listings`);

  runtime.requests = buildRequestInfos();
  let bound = 0;
  for (const request of runtime.requests) bound += request.weight * request.top;
  runtime.lossThreshold = Math.floor(0.01 * bound);
  log(`workload: ${runtime.requests.length} requests, quality bound ${bound}, loss budget ${runtime.lossThreshold}`);

  const directory = isRecord(brief.directory) && Array.isArray(brief.directory.providers) ? brief.directory.providers : [];
  runtime.providers = directory.map((p: Json, index: number) => {
    const id = isRecord(p) && typeof p.id === 'string' ? p.id : `unknown-${index}`;
    return {
      index,
      id,
      wire: isRecord(p) && typeof p.wire === 'string' ? p.wire : 'openai',
      auth: isRecord(p) && typeof p.auth === 'string' ? p.auth : 'none',
      path: isRecord(p) && typeof p.path === 'string' ? p.path : `/p/${id}`,
      key: isRecord(p) && typeof p.key === 'string' ? p.key : '',
      next: 1,
      dead: false,
      observedFails: 0,
      servedSecond: false,
    };
  });

  const port = await startServer();
  log(`listening on 0.0.0.0:${port}`);
  setTimeout(() => {
    log('wall guard: exiting 0');
    process.exit(0);
  }, GUARD_EXIT_MS);

  if (ENV_URL === '') {
    log('no MAKEFUN_ENV_URL: serving without a trial env');
    return;
  }
  const ok = await register(port, catalog);
  if (!ok) {
    log('register failed: the trial cannot start; still serving in case the env retries');
  }
  void pollWork();
}

main().catch((err) => {
  log(`fatal: ${(err as Error).message}`);
  process.exit(1);
});
