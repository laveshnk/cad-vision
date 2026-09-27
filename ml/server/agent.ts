/**
 * Gemini turn loop for the voice agent.
 *
 * One turn is: the user's transcript (plus a summary of the scene and a
 * snapshot of the viewport) goes in, and either spoken text or a batch of tool
 * calls comes out. The browser executes the tools and posts the outcomes back,
 * which continues the same conversation until the model answers with words.
 *
 * Conversation history lives here, keyed by session id, so the browser only
 * ever ships the current turn over the wire.
 */

import { GoogleGenAI } from '@google/genai';
import type { Content, FunctionDeclaration, Part } from '@google/genai';
import { AGENT_SYSTEM_PROMPT, AGENT_TOOLS, describeScene } from '../shared/agentTools.ts';
import type { RawToolCall, SceneSummary, ToolOutcome } from '../shared/agentTools.ts';

export interface TurnRequest {
  sessionId: string;
  /** A committed transcript. Omitted when reporting tool outcomes. */
  text?: string;
  /** Results of the tool calls from the previous response. */
  toolOutcomes?: ToolOutcome[];
  scene: SceneSummary;
  /** Base64 JPEG of the viewport, sent with spoken turns only. */
  snapshotJpegBase64?: string;
}

export interface TurnResponse {
  text: string;
  toolCalls: RawToolCall[];
}

interface Session {
  contents: Content[];
  lastSeen: number;
}

const SESSION_TTL_MS = 30 * 60 * 1000;
/** Keep the tail of the conversation; older turns add latency, not accuracy. */
const MAX_HISTORY_ENTRIES = 24;
/** Upstream statuses that mean "busy, not wrong" — worth another try. */
const RETRY_STATUSES = [429, 500, 503];
/** Backoff between retries. Short: the user is standing there waiting. */
const RETRY_DELAYS_MS = [400, 1_200];

const sessions = new Map<string, Session>();

/**
 * The shared declarations are plain JSON schema so the browser can import them
 * without pulling in the Gemini SDK. Their string types line up with the SDK's
 * `Type` enum values, so this is the one place we assert that.
 */
const tools = [{ functionDeclarations: AGENT_TOOLS as unknown as FunctionDeclaration[] }];

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set');
  if (!client) client = new GoogleGenAI({ apiKey });
  return client;
}

function getSession(sessionId: string): Session {
  sweepExpiredSessions();
  let session = sessions.get(sessionId);
  if (!session) {
    session = { contents: [], lastSeen: Date.now() };
    sessions.set(sessionId, session);
  }
  session.lastSeen = Date.now();
  return session;
}

function sweepExpiredSessions(): void {
  const cutoff = Date.now() - SESSION_TTL_MS;
  for (const [id, session] of sessions) {
    if (session.lastSeen < cutoff) sessions.delete(id);
  }
}

/** Drop the oldest turns, but never start the window on a tool response —
 *  Gemini rejects a function response with no matching call. */
function trimHistory(contents: Content[]): void {
  while (contents.length > MAX_HISTORY_ENTRIES) {
    contents.shift();
    const first = contents[0];
    const startsMidToolCall = first?.parts?.some((part) => 'functionResponse' in part);
    if (startsMidToolCall) contents.shift();
  }
}

/**
 * One `generateContent` call, retried when the failure is worth retrying.
 *
 * Two recoverable cases, both of which would otherwise cost the user the
 * sentence they just said:
 *   - the model is busy (429 / 500 / 503) — wait and ask again;
 *   - the model refuses the viewport snapshot — drop the image and send the
 *     words on their own, since the picture is only ever a hint.
 */
async function generate(contents: Content[]) {
  let lastError: unknown;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      return await getClient().models.generateContent({
        model: process.env.GEMINI_MODEL ?? 'gemini-2.5-flash',
        contents,
        config: {
          systemInstruction: AGENT_SYSTEM_PROMPT,
          tools,
          temperature: 0.7,
        },
      });
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('Unable to process input image') && stripImages(contents)) continue;
      // The SDK carries the upstream status inside the serialized error body.
      const status = /"code"\s*:\s*(\d{3})/.exec(message)?.[1];
      if (!status || !RETRY_STATUSES.includes(Number(status))) throw error;
      const delay = RETRY_DELAYS_MS[attempt];
      if (delay === undefined) break;
      console.warn(`[agent] Gemini returned ${status}; retrying in ${delay}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}

/** Remove image parts from the history. Returns true if any were dropped. */
function stripImages(contents: Content[]): boolean {
  let removed = false;
  for (const content of contents) {
    if (!content.parts) continue;
    const kept = content.parts.filter((part) => !part.inlineData);
    // Never leave a turn with no parts at all — Gemini rejects that outright.
    if (kept.length !== content.parts.length && kept.length > 0) {
      content.parts = kept;
      removed = true;
    }
  }
  return removed;
}

export async function runTurn(request: TurnRequest): Promise<TurnResponse> {
  const session = getSession(request.sessionId);

  if (request.toolOutcomes && request.toolOutcomes.length > 0) {
    session.contents.push({
      role: 'user',
      parts: request.toolOutcomes.map((outcome) => ({
        functionResponse: {
          id: outcome.id,
          name: outcome.name,
          response: { result: outcome.detail, ok: outcome.ok },
        },
      })),
    });
  } else {
    const parts: Part[] = [];
    if (request.snapshotJpegBase64) {
      parts.push({
        inlineData: { mimeType: 'image/jpeg', data: request.snapshotJpegBase64 },
      });
    }
    parts.push({
      text: [
        `[scene] ${describeScene(request.scene)}`,
        `[state] next hand-built shape: ${request.scene.activeShape}; camera ${
          request.scene.cameraRunning ? 'on' : 'off'
        }.`,
        `[user] ${request.text ?? ''}`,
      ].join('\n'),
    });
    session.contents.push({ role: 'user', parts });
  }

  trimHistory(session.contents);

  const response = await generate(session.contents);

  const modelContent = response.candidates?.[0]?.content;
  if (modelContent) session.contents.push(modelContent);

  const toolCalls: RawToolCall[] = (response.functionCalls ?? []).map((call) => ({
    id: call.id,
    name: call.name ?? '',
    args: (call.args ?? {}) as Record<string, unknown>,
  }));

  return { text: response.text ?? '', toolCalls };
}

/** Forget a conversation (called when the user turns voice off). */
export function endSession(sessionId: string): void {
  sessions.delete(sessionId);
}
