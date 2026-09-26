/**
 * Voice backend for the CAD playground.
 *
 * Small on purpose: a plain `node:http` server plus one WebSocket route. It
 * exists because three things cannot live in the browser — the Gemini,
 * ElevenLabs and Presage API keys, and the Presage SDK itself, which has no
 * browser build.
 *
 *   POST /api/voice/token   mint a single-use ElevenLabs Scribe token
 *   POST /api/voice/tts     stream spoken audio for one agent reply
 *   POST /api/agent/turn    run one Gemini turn (text or tool outcomes in)
 *   POST /api/agent/end     forget a conversation
 *   WS   /ws/presage        webcam JPEG frames in, talking flags out
 *   GET  /healthz           which integrations are configured
 *
 * Run it with `npm run dev:server`; Vite proxies /api and /ws to it.
 */

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { Readable } from 'node:stream';
import type { ReadableStream as NodeWebReadableStream } from 'node:stream/web';
import { WebSocketServer } from 'ws';
import { endSession, runTurn } from './agent.ts';
import type { TurnRequest } from './agent.ts';
import { createScribeToken, streamSpeech } from './elevenlabs.ts';
import { createPresageSession } from './presage.ts';
import type { PresageSession } from './presage.ts';

const PORT = Number(process.env.PORT ?? 8787);
/** Viewport snapshots dominate the payload; this is roomy for a JPEG. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

function sendJson(response: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  response.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  });
  response.end(payload);
}

async function readJsonBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new Error('Request body too large');
    chunks.push(chunk as Buffer);
  }
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const server = createServer((request, response) => {
  void handleRequest(request, response).catch((error: unknown) => {
    console.error('[server] unhandled failure:', error);
    if (!response.headersSent) sendJson(response, 500, { error: errorMessage(error) });
    else response.end();
  });
});

async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const { pathname } = new URL(request.url ?? '/', 'http://localhost');

  if (request.method === 'GET' && pathname === '/healthz') {
    sendJson(response, 200, {
      ok: true,
      gemini: Boolean(process.env.GEMINI_API_KEY),
      elevenlabs: Boolean(process.env.ELEVENLABS_API_KEY && process.env.ELEVENLABS_VOICE_ID),
      presage: Boolean(process.env.PRESAGE_API_KEY),
    });
    return;
  }

  if (request.method !== 'POST') {
    sendJson(response, 404, { error: 'Not found' });
    return;
  }

  switch (pathname) {
    case '/api/voice/token': {
      try {
        sendJson(response, 200, { token: await createScribeToken() });
      } catch (error) {
        sendJson(response, 502, { error: errorMessage(error) });
      }
      return;
    }

    case '/api/voice/tts': {
      const body = (await readJsonBody(request)) as { text?: unknown };
      const text = typeof body.text === 'string' ? body.text.trim() : '';
      if (!text) {
        sendJson(response, 400, { error: 'text is required' });
        return;
      }
      try {
        const audio = await streamSpeech(text);
        response.writeHead(200, { 'content-type': 'audio/mpeg', 'cache-control': 'no-store' });
        // The SDK hands back a web stream; Node needs its own Readable to pipe.
        Readable.fromWeb(audio as NodeWebReadableStream<Uint8Array>).pipe(response);
      } catch (error) {
        sendJson(response, 502, { error: errorMessage(error) });
      }
      return;
    }

    case '/api/agent/turn': {
      const body = (await readJsonBody(request)) as Partial<TurnRequest>;
      if (!body.sessionId || !body.scene) {
        sendJson(response, 400, { error: 'sessionId and scene are required' });
        return;
      }
      try {
        sendJson(response, 200, await runTurn(body as TurnRequest));
      } catch (error) {
        sendJson(response, 502, { error: errorMessage(error) });
      }
      return;
    }

    case '/api/agent/end': {
      const body = (await readJsonBody(request)) as { sessionId?: unknown };
      if (typeof body.sessionId === 'string') endSession(body.sessionId);
      sendJson(response, 200, { ok: true });
      return;
    }

    default:
      sendJson(response, 404, { error: 'Not found' });
  }
}

/* ---- Presage frame socket ---- */

const presageSockets = new WebSocketServer({ noServer: true });

presageSockets.on('connection', (socket) => {
  let session: PresageSession | null = null;
  let lastTalking: boolean | null = null;

  const send = (message: unknown) => {
    if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(message));
  };

  createPresageSession(
    (talking) => {
      // Only report edges; the SDK repeats its last detection every frame.
      if (talking === lastTalking) return;
      lastTalking = talking;
      send({ type: 'talking', talking, ts: Date.now() });
    },
    (reason) => {
      // The pipeline is dead. Tell the browser so it falls back to the Talk
      // button instead of streaming frames into a session that rejects them.
      send({ type: 'unavailable', reason });
      socket.close();
    },
  )
    .then((created) => {
      if (socket.readyState !== socket.OPEN) {
        void created.close();
        return;
      }
      session = created;
      send({ type: 'ready' });
    })
    .catch((error: unknown) => {
      console.error('[presage] session failed to start:', error);
      send({ type: 'unavailable', reason: errorMessage(error) });
      socket.close();
    });

  socket.on('message', (data, isBinary) => {
    if (!isBinary || !session) return;
    session.pushFrame(Buffer.isBuffer(data) ? data : Buffer.from(data as ArrayBuffer));
  });

  socket.on('close', () => {
    void session?.close();
    session = null;
  });
});

server.on('upgrade', (request, socket, head) => {
  const { pathname } = new URL(request.url ?? '/', 'http://localhost');
  if (pathname !== '/ws/presage') {
    socket.destroy();
    return;
  }
  presageSockets.handleUpgrade(request, socket, head, (ws) => {
    presageSockets.emit('connection', ws, request);
  });
});

server.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT}`);
  if (!process.env.GEMINI_API_KEY) console.warn('[server] GEMINI_API_KEY missing — /api/agent/turn will fail');
  if (!process.env.ELEVENLABS_API_KEY) console.warn('[server] ELEVENLABS_API_KEY missing — voice in/out will fail');
  if (!process.env.PRESAGE_API_KEY) console.warn('[server] PRESAGE_API_KEY missing — hands-free mic gating is off');
});
