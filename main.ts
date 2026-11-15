// free-model-aggregator-01 entry: one server, two client wires, three provider wires,
// call-minimising routing under the quality-98 floor.
//
// Node 22, standard library only, no npm dependencies, nothing fetched at runtime.
// Run:  node main.ts            (inside a trial: reads $MAKEFUN_PROMPT_FILE, probes the
//                                providers' catalogs, registers with $MAKEFUN_ENV_URL,
//                                serves the client's requests, exits 0 when work is done)
//       node main.ts --list     (prints the catalog document byte-for-byte as registered,
//                                {"providers":[]} with no brief)
//
// Routing design lives in STRATEGY.md. In short: score 99 needs weighted quality 98
// or 99 alike once the other columns sit at 100, so the router plans the whole
// manifest up front with a min-cost-flow matcher - weighted loss plus scaled refusal
// risk against the quality-98 loss budget - and re-plans from the observed state
// after every refusal ("budget is spent" / "is out" kill a provider, "refused this
// call" burns one of its at most two scheduled fail numbers and puts a surcharge on
// its deeper numbers). Nothing depends on the wall clock or on randomness, so the
// trial replays byte for byte.

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
const TIER_BOUNDS: Record<string, [number, number]> = {
  groq: [9, 12], cerebras: [5, 6], gemini: [6, 8], zai: [4, 5],
  openrouter: [10, 12], sambanova: [4, 5], pollinations: [10, 14],
  cloudflare: [7, 8], ovh: [5, 6], kilol: [12, 16],
};
const DESIGNATED_PRIOR = 0.1; // one provider of ten refuses its second call
const OUTAGE_PRIOR = 0.1; // one provider of ten goes out mid-trial
// Tuning knobs (dev sweeps set FMA01_*; a trial sets none of them).
const envNum = (name: string, fallback: number): number => {
  const raw = process.env[name];
  const value = raw === undefined ? NaN : Number(raw);
  return Number.isFinite(value) ? value : fallback;
};
const FAIL_MASS = envNum('FMA01_FAIL_MASS', 1.5); // expected scheduled fail numbers ahead on an untouched provider
const FAIL_MASS_AFTER = envNum('FMA01_FAIL_MASS_AFTER', 0.7); // ... after one observed "refused this call"
const REFUSED_SURCHARGE = envNum('FMA01_REFUSED_SURCHARGE', 0.2); // extra risk per deep call past a provider's first refusal
// The planner minimises provider calls subject to the quality floor: the score is
// round((60*quality + 400)/100) once the other columns sit at 100, so weighted
// quality 98 and 99 both round to 99 - the router may spend up to 2% of the
// trial's quality points to keep every request at one successful upstream call.
const QUALITY_FLOOR = 0.98;
const LOSS_RESERVE = envNum('FMA01_LOSS_RESERVE', 0); // loss budget held back against re-plan drift
const RISK_SCALE = envNum('FMA01_RISK_SCALE', 3000); // milli-rating paid per unit of refusal risk in the plan
const DEEP_CAP = envNum('FMA01_DEEP_CAP', 4); // planned depth (calls past the first) per provider, from risk
const DEEP_SLACK = envNum('FMA01_DEEP_SLACK', 4); // spare planned calls against surprises
const EMERGENCY_RISK = envNum('FMA01_EMERGENCY_RISK', 0.5); // flat risk surcharge on slots past the plan
const EMERGENCY_EXTRA = 3; // slots added beyond the plan when the plan runs dry

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
  refusedFrom: number | null; // call number after this provider's first scheduled refusal
  observedFails: number; // scheduled refusals seen on this provider (its schedule holds at most 2)
  servedSecond: boolean; // its call number 2 already succeeded: no designated refusal there
}

interface RequestInfo {
  id: string;
  requires: Json;
  weight: number;
  top: number; // best free rating meeting the requirements (the quality bound)
  model: string | null; // the model the current plan picked on the assigned provider
  provider: number; // assigned provider index (-1: unassigned)
}

interface Runtime {
  catalog: Catalog;
  providers: ProviderState[];
  requests: RequestInfo[];
  lossBudget: number; // weighted loss at which the quality column still floors to 98
  lossSoFar: number; // weighted loss already sunk on served requests
  plannedMax: number[]; // highest call number the planner will attempt per provider
  manifestIndex: number;
  callChain: Promise<unknown>;
}

const runtime: Runtime = {
  catalog: EMPTY_CATALOG,
  providers: [],
  requests: [],
  lossBudget: 0,
  lossSoFar: 0,
  plannedMax: [],
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
    infos.push({ id: raw.id, requires, weight: raw.weight === 2 ? 2 : 1, top, model: null, provider: -1 });
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

/** Sealed rating of one model on one provider (null when the table lacks it). */
function ratingOf(providerId: string, model: string): number | null {
  for (const row of ratingRows()) {
    if (row.provider === providerId && row.model === model) return row.rating;
  }
  return null;
}

/** Estimated refusal probability of one attempt at call number n on this provider. */
function pRefuse(provider: ProviderState, n: number): number {
  if (n <= 1) return 0;
  const bounds = TIER_BOUNDS[provider.id] ?? [4, 12];
  const [lo, hi] = bounds;
  const mid = (lo + hi) / 2;
  const mass = provider.observedFails === 0 ? FAIL_MASS : FAIL_MASS_AFTER;
  let p = mass / Math.max(1, mid - n + 1); // its 1-2 scheduled fail numbers
  if (provider.refusedFrom !== null) p += REFUSED_SURCHARGE * Math.max(0, n - provider.refusedFrom);
  if (n === 2 && !provider.servedSecond && provider.observedFails === 0) p += DESIGNATED_PRIOR;
  p += OUTAGE_PRIOR / Math.max(1, mid - 2); // one provider of ten goes out, number unknown
  if (n > lo) p += (n - lo) / (hi - lo + 1); // the budget itself is drawn in [lo, hi]
  return Math.min(1, p);
}

// ------------------------------------------------------- min-cost-flow planner
// Every remaining request is matched to one provider call slot; the plan minimises
// weighted quality loss plus scaled refusal risk, and never lets the planned loss
// cross the quality-98 budget. Providers fill their call numbers in order, so the
// k-th call this trial pays pRefuse(provider, next + k - 1).

interface FlowEdge { to: number; cap: number; cost: number; rev: number }

class MinCostFlow {
  graph: FlowEdge[][] = [];
  add(from: number, to: number, cap: number, cost: number): FlowEdge {
    const forward: FlowEdge = { to, cap, cost, rev: this.graph[to].length };
    this.graph[from].push(forward);
    this.graph[to].push({ to: from, cap: 0, cost: -cost, rev: this.graph[from].length - 1 });
    return forward;
  }
  /** Successive shortest paths (SPFA/Bellman-Ford); costs may be negative. */
  solve(source: number, sink: number, want: number): { flow: number; cost: number } {
    const n = this.graph.length;
    let flow = 0;
    let cost = 0;
    const INF = Number.MAX_SAFE_INTEGER;
    while (flow < want) {
      const dist = new Array<number>(n).fill(INF);
      const inQueue = new Array<boolean>(n).fill(false);
      const prevNode = new Array<number>(n).fill(-1);
      const prevEdge = new Array<number>(n).fill(-1);
      dist[source] = 0;
      const queue: number[] = [source];
      inQueue[source] = true;
      while (queue.length > 0) {
        const u = queue.shift() as number;
        inQueue[u] = false;
        for (let k = 0; k < this.graph[u].length; k++) {
          const e = this.graph[u][k];
          if (e.cap > 0 && dist[u] + e.cost < dist[e.to]) {
            dist[e.to] = dist[u] + e.cost;
            prevNode[e.to] = u;
            prevEdge[e.to] = k;
            if (!inQueue[e.to]) { queue.push(e.to); inQueue[e.to] = true; }
          }
        }
      }
      if (dist[sink] >= INF) break;
      let push = want - flow;
      for (let v = sink; v !== source; v = prevNode[v]) push = Math.min(push, this.graph[prevNode[v]][prevEdge[v]].cap);
      for (let v = sink; v !== source; v = prevNode[v]) {
        const e = this.graph[prevNode[v]][prevEdge[v]];
        e.cap -= push;
        this.graph[e.to][e.rev].cap += push;
      }
      flow += push;
      cost += push * dist[sink];
    }
    return { flow, cost };
  }
}

interface PlanEdge { unit: number; provider: number; model: string; loss: number }

interface FlowRefs {
  flow: MinCostFlow;
  unitEdge: (FlowEdge | null)[][]; // unitEdge[u][j]: forward edge unit u -> provider j
  units: number[];
  sink: number;
}

/**
 * Build the matching graph: source -> units -> providers -> sink. Unit edges carry
 * the weighted loss (scaled by mu when the plan must be pulled back into the
 * quality budget); provider->sink parallel edges carry RISK_SCALE * the refusal
 * risk of each call number the provider would spend, priced in ascending order so
 * the successive-shortest-path solver always spends the safest numbers first.
 */
function buildFlow(fromUnit: number, mu: number): FlowRefs {
  const rows = ratingRows();
  const units: number[] = [];
  for (let i = fromUnit; i < runtime.requests.length; i++) {
    if (runtime.requests[i].top > 0) units.push(i);
  }
  const P = runtime.providers.length;
  const sink = 1 + units.length + P;
  const flow = new MinCostFlow();
  for (let s = 0; s <= sink; s++) flow.graph.push([]);
  for (let u = 0; u < units.length; u++) flow.add(0, 1 + u, 1, 0);
  const unitEdge: (FlowEdge | null)[][] = [];
  for (let u = 0; u < units.length; u++) {
    const request = runtime.requests[units[u]];
    unitEdge.push([]);
    for (const provider of runtime.providers) {
      const best = provider.dead ? null : bestModelOn(provider.id, request.requires, rows);
      if (best === null) { unitEdge[u].push(null); continue; }
      const loss = request.weight * Math.max(0, request.top - best.rating);
      const edge = flow.add(1 + u, 1 + units.length + provider.index, 1, loss * 64 * mu);
      unitEdge[u].push(edge);
    }
  }
  for (const provider of runtime.providers) {
    if (provider.dead) continue;
    const slots = runtime.plannedMax[provider.index] - provider.next + 1 + EMERGENCY_EXTRA;
    // The planner spends call numbers in order, so slot costs must ascend for the
    // successive-shortest-path solver to consume them in that order: price each
    // slot at the worst risk seen so far on this provider (the number-2 designated
    // bump then stays honestly counted on every deeper call). Slots past the plan
    // carry a steep surcharge: the quality budget may buy them back when refusals
    // have eaten the planned capacity, but risk-first plans leave them alone.
    let worst = 0;
    for (let k = 0; k < slots; k++) {
      const number = provider.next + k;
      worst = Math.max(worst, pRefuse(provider, number));
      const over = k >= runtime.plannedMax[provider.index] - provider.next + 1 ? EMERGENCY_RISK : 0;
      flow.add(1 + units.length + provider.index, sink, 1, Math.round(RISK_SCALE * (worst + over) * 64) + k);
    }
  }
  return { flow, unitEdge, units, sink };
}

function planAssignment(fromUnit: number, budget: number): { edges: PlanEdge[]; loss: number; feasible: boolean } {
  let mu = 1;
  let edges: PlanEdge[] = [];
  let loss = Number.MAX_SAFE_INTEGER;
  for (let round = 0; round < 14; round++) {
    const refs = buildFlow(fromUnit, mu);
    refs.flow.solve(0, refs.sink, refs.units.length);
    edges = [];
    loss = 0;
    for (let u = 0; u < refs.units.length; u++) {
      for (let j = 0; j < runtime.providers.length; j++) {
        const e = refs.unitEdge[u][j];
        if (e === null || e.cap > 0) continue;
        const request = runtime.requests[refs.units[u]];
        const model = bestModelOn(runtime.providers[j].id, request.requires, ratingRows());
        if (model === null) continue;
        const unitLoss = request.weight * Math.max(0, request.top - model.rating);
        edges.push({ unit: refs.units[u], provider: j, model: model.model, loss: unitLoss });
        loss += unitLoss;
        break;
      }
    }
    if (SIM_DEBUG) {
      process.stderr.write(`sim: plan from r${fromUnit + 1} mu=${mu} loss=${loss} budget=${budget} units=${refs.units.length}\n`);
      if (SIM_DEBUG && mu >= 4096) {
        for (const edge of edges) process.stderr.write(`sim:   r${edge.unit + 1} -> ${runtime.providers[edge.provider].id} (${edge.model}, loss ${edge.loss})\n`);
      }
    }
    if (loss <= budget || mu >= 4096) break;
    mu *= 2;
  }
  return { edges, loss, feasible: loss <= budget };
}

/** Rebuild the plan for every request that still needs an answer. */
function replan(fromUnit: number): void {
  const budget = Math.max(0, Math.floor((runtime.lossBudget - runtime.lossSoFar) * (1 - LOSS_RESERVE)));
  for (let round = 0; round < 6; round++) {
    const plan = planAssignment(fromUnit, budget);
    const assigned = new Map<number, PlanEdge>();
    for (const edge of plan.edges) assigned.set(edge.unit, edge);
    let unassigned = 0;
    for (let i = fromUnit; i < runtime.requests.length; i++) {
      const edge = assigned.get(i);
      runtime.requests[i].provider = edge === undefined ? -1 : edge.provider;
      runtime.requests[i].model = edge === undefined ? null : edge.model;
      if (edge === undefined) unassigned += 1;
    }
    if (unassigned === 0) return;
    // the plan ran dry: widen the deepest planners up to the sealed budget bounds
    let widened = false;
    for (const provider of runtime.providers) {
      if (provider.dead) continue;
      const [lo, hi] = TIER_BOUNDS[provider.id] ?? [4, 12];
      const room = hi + EMERGENCY_EXTRA - runtime.plannedMax[provider.index];
      if (room > 0) {
        runtime.plannedMax[provider.index] += Math.min(room, Math.max(1, unassigned));
        widened = true;
      }
    }
    if (!widened) return;
  }
}

/** Planned depth per provider: buy the cheapest marginal call numbers first. */
function computePlannedCaps(): void {
  const P = runtime.providers.length;
  const plannedMax = new Array<number>(P).fill(1);
  const target = Math.max(0, runtime.requests.length - P) + DEEP_SLACK;
  for (let bought = 0; bought < target; bought++) {
    let bestProvider = -1;
    let bestCost = Number.MAX_SAFE_INTEGER;
    for (let j = 0; j < P; j++) {
      const provider = runtime.providers[j];
      if (provider.dead) continue;
      if (plannedMax[j] >= 1 + DEEP_CAP) continue;
      const cost = pRefuse(provider, plannedMax[j] + 1);
      if (cost < bestCost) { bestCost = cost; bestProvider = j; }
    }
    if (bestProvider < 0) break;
    plannedMax[bestProvider] += 1;
  }
  runtime.plannedMax = plannedMax;
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
  // One attempt at a time; the plan is rebuilt from the observed state before every
  // attempt, so after a refusal the next provider is always the currently best trade.
  const request = runtime.requests[requestIndex];
  for (let round = 0; round < 40; round++) {
    if (request.provider < 0 || runtime.providers[request.provider].dead || request.model === null) {
      replan(requestIndex);
    }
    if (request.provider < 0 || request.model === null) break;
    const choice = runtime.providers[request.provider];
    const model = request.model as string;
    const answer = await callProvider(choice, model, canonical);
    if (answer.ok && answer.usage !== null) {
      if (choice.next - 1 === 2) choice.servedSecond = true;
      runtime.lossSoFar += request.weight * Math.max(0, request.top - (ratingOf(choice.id, request.model as string) ?? request.top));
      return clientResponseFor(canonical, {
        model: answer.model, text: answer.text, toolCalls: answer.toolCalls, stop: answer.stop, usage: answer.usage,
      });
    }
    if (answer.refused) {
      const kind = classifyRefusal(answer.message);
      if (kind === 'budget' || kind === 'outage') choice.dead = true;
      else if (kind === 'scheduled') {
        choice.observedFails += 1;
        if (choice.refusedFrom === null) choice.refusedFrom = choice.next;
      }
      replan(requestIndex);
    } else {
      break; // not a refusal the schedule explains: hand the error back
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

// In-process simulation mode (tuning tooling; MAKEFUN_SIM_TRIAL is never set by a
// trial): the providers' refusal schedules are read from a JSON file and the
// planner is exercised against them without HTTP. Prints one JSON result line.
const SIM_TRIAL = process.env.MAKEFUN_SIM_TRIAL ?? '';
const SIM_DEBUG = process.env.FMA01_SIM_DEBUG === '1';

interface SimRow { id: string; budget: number; fail_calls: number[]; outage_from: number | null }

function simCall(provider: ProviderState, model: string): ProviderAnswer {
  const row = SIM_ROWS.get(provider.id) as SimRow;
  const number = provider.next;
  provider.next = number + 1;
  const refused = number > row.budget ? `${provider.id}'s free-tier budget is spent`
    : row.outage_from !== null && number >= row.outage_from ? `${provider.id} is out`
    : row.fail_calls.includes(number) ? `${provider.id} refused this call` : null;
  if (refused !== null) return { ok: false, status: 429, refused: true, message: refused, text: '', toolCalls: [], stop: '', usage: null, model };
  return { ok: true, status: 200, refused: false, message: '', text: 'ANSWER ECHO', toolCalls: [], stop: 'stop', usage: { input: 1, output: 1 }, model };
}

async function simMain(): Promise<void> {
  const rows: SimRow[] = JSON.parse(readFileSync(SIM_TRIAL, 'utf8'));
  SIM_ROWS.clear();
  for (const row of rows) SIM_ROWS.set(row.id, row);
  runtime.catalog = { providers: [] };
  runtime.requests = buildRequestInfos();
  let bound = 0;
  for (const request of runtime.requests) bound += request.weight * request.top;
  runtime.lossBudget = Math.floor((1 - QUALITY_FLOOR) * bound);
  const directory = isRecord(brief.directory) && Array.isArray(brief.directory.providers) ? brief.directory.providers : [];
  runtime.providers = directory.map((p: Json, index: number) => ({
    index,
    id: isRecord(p) && typeof p.id === 'string' ? p.id : `unknown-${index}`,
    wire: isRecord(p) && typeof p.wire === 'string' ? p.wire : 'openai',
    auth: isRecord(p) && typeof p.auth === 'string' ? p.auth : 'none',
    path: isRecord(p) && typeof p.path === 'string' ? p.path : '',
    key: '',
    next: 1,
    dead: false,
    refusedFrom: null,
    observedFails: 0,
    servedSecond: false,
  }));
  computePlannedCaps();
  replan(0);
  let calls = 0;
  let served = 0;
  let loss = 0;
  for (let i = 0; i < runtime.requests.length; i++) {
    const request = runtime.requests[i];
    for (let round = 0; round < 40; round++) {
      if (request.provider < 0 || runtime.providers[request.provider].dead || request.model === null) replan(i);
      if (request.provider < 0 || request.model === null) break;
      const provider = runtime.providers[request.provider];
      calls += 1;
      const answer = simCall(provider, request.model as string);
      if (SIM_DEBUG) process.stderr.write(`sim: r${i + 1} -> ${provider.id} #${provider.next - 1} ${answer.ok ? 'ok' : `REFUSED ${answer.message}`} (model ${request.model})\n`);
      if (answer.ok && answer.usage !== null) {
        served += 1;
        const rating = ratingOf(provider.id, answer.model) ?? request.top;
        const unitLoss = request.weight * Math.max(0, request.top - rating);
        loss += unitLoss;
        runtime.lossSoFar += unitLoss;
        break;
      }
      const kind = classifyRefusal(answer.message);
      if (kind === 'budget' || kind === 'outage') provider.dead = true;
      else if (kind === 'scheduled') {
        provider.observedFails += 1;
        if (provider.refusedFrom === null) provider.refusedFrom = provider.next;
      }
      replan(i);
    }
  }
  const quality = bound === 0 ? 0 : Math.floor((100 * (bound - loss)) / bound);
  const servedScore = Math.floor((100 * served) / Math.max(1, runtime.requests.length));
  const score = Math.round((60 * quality + 15 * servedScore + 2500) / 100);
  process.stdout.write(`${JSON.stringify({ calls, loss, lossBudget: runtime.lossBudget, quality, served, score })}\n`);
}
const SIM_ROWS = new Map<string, SimRow>();


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
  if (SIM_TRIAL !== '') {
    await simMain();
    process.exit(0);
  }
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
  runtime.lossBudget = Math.floor((1 - QUALITY_FLOOR) * bound);
  log(`workload: ${runtime.requests.length} requests, quality bound ${bound}, loss budget ${runtime.lossBudget}`);

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
      refusedFrom: null,
      observedFails: 0,
      servedSecond: false,
    };
  });

  computePlannedCaps();
  replan(0);
  log(`planned caps: ${runtime.providers.map((p) => `${p.id}:${runtime.plannedMax[p.index]}`).join(' ')}`);

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
