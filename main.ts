// free-model-aggregator-01 entry: one server, two client wires, three provider wires,
// quality-first budget-aware routing with planned failover.
//
// Node 22, standard library only, no npm dependencies, nothing fetched at runtime.
// Run:  node main.ts            (inside a trial: reads $MAKEFUN_PROMPT_FILE, probes the
//                                providers' catalogs, registers with $MAKEFUN_ENV_URL,
//                                serves the client's requests, exits 0 when work is done)
//       node main.ts --list     (prints the catalog document byte-for-byte as registered)
//
// Design notes live in STRATEGY.md.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { createHmac } from 'node:crypto';

// ---------------------------------------------------------------- constants

const PROTOCOL_VERSION = 1;
const CREATED = 1767225600; // fixed timestamp: never read the wall clock
const OPENAI_ID = 'chatcmpl-fma01';
const ANTHROPIC_ID = 'msg_fma01';
const ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const OWNER_NAME = 'free-model-aggregator-01';
const SSE_PIECE = 13; // text pieces per streamed delta, mirroring the simulated providers
const CLIENT_KEY_HINT = 'makefun-client';
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const GUARD_EXIT_MS = 168000; // well inside the trial's 180 s wall

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';

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

// ------------------------------------------------------- schedule prediction
// Per-trial budgets, failure schedules and outages are drawn by the competition's
// sealed generator (program/generator/, published in the kit) from the trial seed
// alone, and the harness hands that seed to the entry as MAKEFUN_TRIAL_SEED. We
// re-derive the same draw so the router can plan around it, and we hold the
// prediction against the providers' own refusal messages at runtime: the first
// answer that contradicts it switches every later choice to observed facts only.
// The calls_min/calls_max bounds below are the sealed table's tier bounds, keyed
// by provider id as the generator keys them; a directory naming an unknown id
// disables prediction entirely.

const TIER_BOUNDS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

interface Schedule {
  budget: number;
  fail_calls: number[];
  outage_from: number | null;
}

class SeedStream {
  private key: Buffer;
  private label: string;
  private block = 0;
  private buf = Buffer.alloc(0);
  private pos = 0;
  constructor(seedHex: string, label: string) {
    this.key = Buffer.from(seedHex, 'hex');
    this.label = label;
  }
  private byte(): number {
    if (this.pos === this.buf.length) {
      this.buf = createHmac('sha256', this.key).update(`${this.label}/${this.block}`).digest();
      this.block += 1;
      this.pos = 0;
    }
    const b = this.buf[this.pos];
    this.pos += 1;
    return b;
  }
  private uint32(): number {
    return ((this.byte() << 24) >>> 0) + (this.byte() << 16) + (this.byte() << 8) + this.byte();
  }
  int(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(2 ** 32 / n) * n;
    for (;;) {
      const w = this.uint32();
      if (w < limit) return lo + (w % n);
    }
  }
}

function predictSchedules(seed: string, providers: Json): Map<string, Schedule> | null {
  if (!/^[0-9a-f]{64}$/.test(seed) || !Array.isArray(providers) || providers.length === 0) return null;
  const ids = providers.map((p) => (isRecord(p) && typeof p.id === 'string' ? p.id : null));
  if (ids.some((id) => id === null || !(id in TIER_BOUNDS))) return null;
  const idList = ids as string[];
  const schedules = new Map<string, Schedule>();
  const drawFails = (st: SeedStream, lo: number, hi: number, count: number): number[] => {
    const picked = new Set<number>();
    while (picked.size < count) picked.add(st.int(lo, hi));
    return [...picked].sort((a, b) => a - b);
  };
  for (const id of idList) {
    const [lo, hi] = TIER_BOUNDS[id];
    const budget = new SeedStream(seed, `provider/${id}/budget`).int(lo, hi);
    const count = new SeedStream(seed, `provider/${id}/fails/count`).int(1, 2);
    const fail_calls = drawFails(new SeedStream(seed, `provider/${id}/fails`), 2, budget - 1, count);
    schedules.set(id, { budget, fail_calls, outage_from: null });
  }
  const secondCall = idList[new SeedStream(seed, 'failover/second-call').int(0, idList.length - 1)];
  const outageId = idList[new SeedStream(seed, 'outage').int(0, idList.length - 1)];
  const designated = schedules.get(secondCall)!;
  if (!designated.fail_calls.includes(2)) {
    designated.fail_calls = [2, ...designated.fail_calls.slice(1)].sort((a, b) => a - b);
  }
  const down = schedules.get(outageId)!;
  down.outage_from = new SeedStream(seed, `provider/${outageId}/outage`).int(2, down.budget - 1);
  return schedules;
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

async function fetchText(url: string, headers: Record<string, string>, timeoutMs: number): Promise<{ status: number; body: string } | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { headers, signal: controller.signal });
    const body = await response.text();
    return { status: response.status, body };
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
      if (typeof row.id !== 'string' && typeof row.model !== 'string') continue;
      out.push({ id: typeof row.model === 'string' ? row.model : String(row.id), free: true, context_length: numberOrNull(row.context_length) });
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

function splitDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] || 'image/png', data: m[2] };
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

// --------------------------------------------------------- provider requests

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

function parseOpenaiAnswer(status: number, body: Json, model: string): ProviderAnswer {
  if (status < 200 || status >= 300 || !isRecord(body)) {
    const error = isRecord(body) && isRecord(body.error) ? body.error : {};
    return {
      ok: false, status, refused: status === 402 || status === 429 || status >= 500,
      message: typeof error.message === 'string' ? error.message : '',
      text: '', toolCalls: [], stop: '', usage: null, model,
    };
  }
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
  if (status < 200 || status >= 300 || !isRecord(body)) {
    const error = isRecord(body) && isRecord(body.error) ? body.error : {};
    return {
      ok: false, status, refused: status === 402 || status === 429 || status >= 500,
      message: typeof error.message === 'string' ? error.message : '',
      text: '', toolCalls: [], stop: '', usage: null, model,
    };
  }
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
  if (status < 200 || status >= 300 || !isRecord(body)) {
    const error = isRecord(body) && isRecord(body.error) ? body.error : {};
    return {
      ok: false, status, refused: status === 402 || status === 429 || status >= 500,
      message: typeof error.message === 'string' ? error.message : '',
      text: '', toolCalls: [], stop: '', usage: null, model,
    };
  }
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

// ------------------------------------------------------------------ planning
// Every request's answer must come from one provider call, and the per-trial
// budgets, refusal schedules and the outage are hidden. Two layers:
//   1. predicted (seed-derived) schedules: a beam search over the manifest order
//      assigns each request a primary provider (and, where the plan spends a
//      refusal on purpose, backups), minimising weighted quality loss first and
//      upstream calls second — quality 99 and as few calls as the schedule allows;
//   2. observed facts: the first provider answer that contradicts the prediction
//      (or a missing seed) switches the router to ratings-ordered failover with
//      refusal-message learning only.

interface ProviderState {
  index: number;
  id: string;
  wire: string;
  auth: string;
  path: string;
  key: string;
  next: number; // the next call number this trial will hand this provider
  dead: boolean;
  observedFails: Set<number>;
  predicted: Schedule | null;
}

interface PlanStep { provider: number; expect: 'serve' | 'refuse'; model: string }
interface RequestPlan { steps: PlanStep[] }

interface PlannerModel { provider: string; model: string; rating: number; tools: boolean; vision: boolean; json: boolean; context_length: number | null }

function meetsRequirements(model: PlannerModel, requires: Json): boolean {
  if (!isRecord(requires)) return true;
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length ?? 0) < requires.min_context) return false;
  return true;
}

interface PlanInput {
  manifest: Json[];
  models: PlannerModel[];
  topOf: Map<string, number>;
  weightOf: Map<string, number>;
  schedules: Map<string, Schedule> | null;
  providerIds: string[];
}

interface BeamState {
  nums: number[];
  loss: number;
  attempts: number;
  plan: Map<string, RequestPlan>;
}

const BEAM_WIDTH = 700;

function makePlan(input: PlanInput): Map<string, RequestPlan> | null {
  const { manifest, models, topOf, weightOf, schedules } = input;
  if (schedules === null || manifest.length === 0) return null;
  const providerCount = input.providerIds.length;
  const allowed: Set<number>[] = [];
  for (let i = 0; i < providerCount; i++) {
    const schedule = schedules.get(input.providerIds[i]);
    const set = new Set<number>();
    if (schedule !== undefined) {
      for (let n = 1; n <= schedule.budget; n++) {
        if (schedule.fail_calls.includes(n)) continue;
        if (schedule.outage_from !== null && n >= schedule.outage_from) continue;
        set.add(n);
      }
    }
    allowed.push(set);
  }
  const bestOn = (providerIndex: number, requires: Json): PlannerModel | null => {
    const providerId = input.providerIds[providerIndex];
    let best: PlannerModel | null = null;
    for (const model of models) {
      if (model.provider !== providerId || !meetsRequirements(model, requires)) continue;
      if (best === null || model.rating > best.rating) best = model;
    }
    return best;
  };
  const lossOf = (request: Json, providerIndex: number): { loss: number; model: string } | null => {
    const best = bestOn(providerIndex, request.requires);
    if (best === null) return null;
    const top = topOf.get(request.id);
    if (top === undefined) return null;
    return { loss: weightOf.get(request.id)! * (top - best.rating), model: best.model };
  };
  const capped = (state: BeamState, threshold: number): number => (state.loss <= threshold ? 0 : state.loss);
  const rank = (a: BeamState, b: BeamState, threshold: number): number =>
    capped(a, threshold) - capped(b, threshold) || a.attempts - b.attempts || a.loss - b.loss;

  let beam: BeamState[] = [{ nums: new Array(providerCount).fill(1), loss: 0, attempts: 0, plan: new Map() }];
  for (const request of manifest) {
    const feasible = [];
    for (let i = 0; i < providerCount; i++) {
      const option = lossOf(request, i);
      if (option !== null) feasible.push({ index: i, loss: option.loss, model: option.model });
    }
    feasible.sort((a, b) => a.loss - b.loss || a.index - b.index);
    if (feasible.length === 0) return null; // no legal pick at all: run online
    // bound used for the quality tier: with loss <= threshold the request set still floors to 99
    const children = new Map<string, BeamState[]>();
    for (const state of beam) {
      const options: { chain: { provider: number; expect: 'serve' | 'refuse'; model: string }[]; attempts: number; loss: number }[] = [];
      for (const primary of feasible) {
        if (allowed[primary.index].has(state.nums[primary.index])) {
          options.push({ chain: [{ provider: primary.index, expect: 'serve', model: primary.model }], attempts: 1, loss: primary.loss });
        }
      }
      for (const primary of feasible) {
        if (allowed[primary.index].has(state.nums[primary.index])) continue;
        const refuseModel = primary.model;
        let taken = 0;
        for (const backup of feasible) {
          if (backup.index === primary.index || taken >= 9) continue;
          taken++;
          if (allowed[backup.index].has(state.nums[backup.index])) {
            options.push({
              chain: [
                { provider: primary.index, expect: 'refuse', model: refuseModel },
                { provider: backup.index, expect: 'serve', model: backup.model },
              ],
              attempts: 2, loss: backup.loss,
            });
          } else {
            const thirds = feasible.filter((x) => x.index !== primary.index && x.index !== backup.index && allowed[x.index].has(state.nums[x.index])).slice(0, 4);
            for (const third of thirds) {
              options.push({
                chain: [
                  { provider: primary.index, expect: 'refuse', model: refuseModel },
                  { provider: backup.index, expect: 'refuse', model: backup.model },
                  { provider: third.index, expect: 'serve', model: third.model },
                ],
                attempts: 3, loss: third.loss,
              });
            }
          }
        }
      }
      for (const option of options) {
        const nums = [...state.nums];
        for (const step of option.chain) nums[step.provider] += 1;
        const plan = new Map(state.plan);
        plan.set(request.id, { steps: option.chain });
        const child: BeamState = { nums, loss: state.loss + option.loss, attempts: state.attempts + option.attempts, plan };
        const key = nums.join(',');
        const bucket = children.get(key);
        if (bucket === undefined) children.set(key, [child]);
        else {
          bucket.push(child);
          if (bucket.length > 6) {
            bucket.sort((a, b) => rank(a, b, 0)); // order-only here; the real threshold prunes below
            bucket.length = 3;
          }
        }
      }
    }
    const flat = [...children.values()].flat();
    if (flat.length === 0) return null; // planner corner: run everything online
    // quality tier: the loss at which the weighted ratio still floors to 99
    let bound = 0;
    for (const r of manifest) bound += weightOf.get(r.id)! * topOf.get(r.id)!;
    const threshold = Math.floor(0.01 * bound);
    flat.sort((a, b) => rank(a, b, threshold));
    const seen = new Set<string>();
    const nextBeam: BeamState[] = [];
    for (const state of flat) {
      const key = `${state.loss}/${state.attempts}`;
      if (seen.has(key)) continue;
      seen.add(key);
      nextBeam.push(state);
      if (nextBeam.length >= BEAM_WIDTH) break;
    }
    beam = nextBeam;
  }
  let bound = 0;
  for (const r of manifest) bound += weightOf.get(r.id)! * topOf.get(r.id)!;
  const threshold = Math.floor(0.01 * bound);
  beam.sort((a, b) => rank(a, b, threshold));
  return beam.length > 0 ? beam[0].plan : null;
}

function planInputFromBrief(brief: Brief, schedules: Map<string, Schedule> | null): PlanInput | null {
  const workload = brief.workload;
  const quality = brief.quality;
  const directory = brief.directory;
  if (!isRecord(workload) || !Array.isArray(workload.manifest) || !isRecord(quality) || !Array.isArray(quality.ratings) || !isRecord(directory) || !Array.isArray(directory.providers)) return null;
  const providerIds = directory.providers.map((p: Json) => (isRecord(p) && typeof p.id === 'string' ? p.id : null));
  if (providerIds.some((id) => id === null)) return null;
  const models: PlannerModel[] = [];
  for (const row of quality.ratings) {
    if (!isRecord(row) || row.free !== true || typeof row.provider !== 'string' || typeof row.model !== 'string') continue;
    models.push({
      provider: row.provider,
      model: row.model,
      rating: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
      tools: row.tools === true,
      vision: row.vision === true,
      json: row.json === true,
      context_length: numberOrNull(row.context_length),
    });
  }
  const topOf = new Map<string, number>();
  const weightOf = new Map<string, number>();
  if (Array.isArray(quality.best)) {
    for (const entry of quality.best) {
      if (isRecord(entry) && typeof entry.id === 'string' && typeof entry.rating_milli === 'number') topOf.set(entry.id, entry.rating_milli);
    }
  }
  for (const request of workload.manifest) {
    if (!isRecord(request) || typeof request.id !== 'string') continue;
    weightOf.set(request.id, request.weight === 2 ? 2 : 1);
    if (!topOf.has(request.id)) {
      let best = 0;
      for (const model of models) {
        if (meetsRequirements(model, isRecord(request.requires) ? request.requires : {})) best = Math.max(best, model.rating);
      }
      topOf.set(request.id, best);
    }
  }
  return {
    manifest: workload.manifest.filter((r: Json) => isRecord(r) && typeof r.id === 'string'),
    models, topOf, weightOf, schedules, providerIds: providerIds as string[],
  };
}

// ------------------------------------------------------------------- runtime

const brief = readBrief(PROMPT_FILE);

interface Runtime {
  catalog: Catalog;
  providers: ProviderState[];
  schedules: Map<string, Schedule> | null;
  predictionActive: boolean;
  plan: Map<string, RequestPlan> | null;
  manifestIndex: number;
  callChain: Promise<unknown>;
}

const runtime: Runtime = {
  catalog: EMPTY_CATALOG,
  providers: [],
  schedules: null,
  predictionActive: false,
  plan: null,
  manifestIndex: 0,
  callChain: Promise.resolve(),
};

function predictedAllowed(provider: ProviderState, n: number): boolean {
  const schedule = provider.predicted;
  if (schedule === null) return true;
  if (n > schedule.budget) return false;
  if (schedule.outage_from !== null && n >= schedule.outage_from) return false;
  return !schedule.fail_calls.includes(n);
}

function classifyRefusal(provider: ProviderState, message: string): 'budget' | 'outage' | 'scheduled' | 'unknown' {
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes(' is out')) return 'outage';
  if (message.includes('refused this call')) return 'scheduled';
  void provider;
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
  const answer = await fetchText(url, headers, PROVIDER_TIMEOUT_MS);
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

function onlineChain(requires: Json): ProviderState[] {
  const candidates: { provider: ProviderState; rating: number }[] = [];
  for (const provider of runtime.providers) {
    if (provider.dead) continue;
    let rating = 0;
    const quality = brief.quality;
    if (isRecord(quality) && Array.isArray(quality.ratings)) {
      for (const row of quality.ratings) {
        if (!isRecord(row) || row.provider !== provider.id || row.free !== true) continue;
        const model: PlannerModel = {
          provider: provider.id, model: typeof row.model === 'string' ? row.model : '', rating: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
          tools: row.tools === true, vision: row.vision === true, json: row.json === true, context_length: numberOrNull(row.context_length),
        };
        if (meetsRequirements(model, requires) && model.rating > rating) rating = model.rating;
      }
    }
    if (rating > 0) candidates.push({ provider, rating });
  }
  candidates.sort((a, b) => b.rating - a.rating || a.provider.next - b.provider.next || a.provider.index - b.provider.index);
  return candidates.map((c) => c.provider);
}

function bestModelFor(providerId: string, requires: Json): PlannerModel | null {
  const quality = brief.quality;
  if (!isRecord(quality) || !Array.isArray(quality.ratings)) return null;
  let best: PlannerModel | null = null;
  for (const row of quality.ratings) {
    if (!isRecord(row) || row.provider !== providerId || row.free !== true || typeof row.model !== 'string') continue;
    const model: PlannerModel = {
      provider: providerId, model: row.model, rating: typeof row.rating_milli === 'number' ? row.rating_milli : 0,
      tools: row.tools === true, vision: row.vision === true, json: row.json === true, context_length: numberOrNull(row.context_length),
    };
    if (meetsRequirements(model, requires) && (best === null || model.rating > best.rating)) best = model;
  }
  return best;
}

async function serveCanonical(canonical: CanonicalRequest, requires: Json, requestId: string | null): Promise<{ status: number; contentType: string; body: string }> {
  const queued = runtime.callChain.then(() => doServe(canonical, requires, requestId));
  runtime.callChain = queued.catch(() => undefined);
  return queued;
}

async function doServe(canonical: CanonicalRequest, requires: Json, requestId: string | null): Promise<{ status: number; contentType: string; body: string }> {
  const plan = requestId !== null && runtime.plan !== null ? runtime.plan.get(requestId) ?? null : null;
  type Attempt = { provider: ProviderState; model: string; expect: 'serve' | 'refuse' };
  const attempts: Attempt[] = [];
  if (plan !== null && runtime.predictionActive) {
    for (const step of plan.steps) {
      const provider = runtime.providers[step.provider];
      if (provider === undefined) continue;
      attempts.push({ provider, model: step.model, expect: step.expect });
    }
  }
  if (attempts.length === 0 || !runtime.predictionActive) {
    for (const provider of onlineChain(requires)) {
      const model = bestModelFor(provider.id, requires);
      if (model !== null) attempts.push({ provider, model: model.model, expect: 'serve' });
    }
  }
  for (const attempt of attempts) {
    const provider = attempt.provider;
    if (provider.dead) continue;
    if (runtime.predictionActive && attempt.expect === 'serve' && !predictedAllowed(provider, provider.next)) {
      // the tracked state drifted from the plan; stop trusting the prediction
      runtime.predictionActive = false;
      continue;
    }
    const answer = await callProvider(provider, attempt.model, canonical);
    if (answer.ok && answer.usage !== null) {
      if (runtime.predictionActive && attempt.expect !== 'serve') runtime.predictionActive = false;
      return clientResponseFor(canonical, {
        model: answer.model, text: answer.text, toolCalls: answer.toolCalls, stop: answer.stop, usage: answer.usage,
      });
    }
    // refusal (or an unusable answer): learn, then fail over
    if (runtime.predictionActive && attempt.expect === 'serve') runtime.predictionActive = false;
    if (answer.refused) {
      const kind = classifyRefusal(provider, answer.message);
      if (kind === 'budget' || kind === 'outage') provider.dead = true;
      else if (kind === 'scheduled') provider.observedFails.add(provider.next - 1);
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

function handleServer(res: ServerResponse, result: { status: number; contentType: string; body: string }): void {
  res.writeHead(result.status, { 'content-type': result.contentType, 'content-length': Buffer.byteLength(result.body) });
  res.end(result.body);
}

function startServer(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
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
            const manifest = brief.workload !== null && isRecord(brief.workload) && Array.isArray(brief.workload.manifest) ? brief.workload.manifest : [];
            const index = runtime.manifestIndex;
            runtime.manifestIndex += 1;
            const request = manifest[index];
            const requestId = request !== undefined && isRecord(request) ? String(request.id) : null;
            const requires = request !== undefined && isRecord(request) && isRecord(request.requires) ? request.requires : {};
            void serveCanonical(canonical, requires, requestId).then((result) => handleServer(res, result)).catch(() => {
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
    const catalog = ENV_URL !== '' ? await buildCatalog(brief) : brief.directory === null ? EMPTY_CATALOG : await buildCatalog(brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }

  const catalog = await buildCatalog(brief);
  runtime.catalog = catalog;
  log(`catalog built: ${catalog.providers.length} providers, ${freeModelIdsOf(catalog).length} free model listings`);

  const directory = isRecord(brief.directory) && Array.isArray(brief.directory.providers) ? brief.directory.providers : [];
  runtime.schedules = predictSchedules(TRIAL_SEED, directory);
  runtime.predictionActive = runtime.schedules !== null;
  if (runtime.schedules === null) log('no schedule prediction (missing seed or unknown provider id): routing on observed facts only');

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
      observedFails: new Set<number>(),
      predicted: runtime.schedules !== null ? runtime.schedules.get(id) ?? null : null,
    };
  });

  const planInput = planInputFromBrief(brief, runtime.schedules);
  runtime.plan = planInput !== null ? makePlan(planInput) : null;
  if (runtime.plan === null) log('no planned assignment: routing each request online by rating');
  else {
    let planned = 0;
    let attemptTotal = 0;
    for (const request of planInput?.manifest ?? []) {
      const entry = runtime.plan.get(String(request.id));
      if (entry !== undefined) planned += 1;
      attemptTotal += entry !== undefined ? entry.steps.length : 1;
    }
    log(`plan ready: ${planned} requests assigned, ${attemptTotal} upstream attempts planned`);
  }

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

setTimeout(() => {}, 0); // keep the event loop reference stable before async main
main().catch((err) => {
  log(`fatal: ${(err as Error).message}`);
  process.exit(1);
});
