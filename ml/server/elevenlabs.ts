/**
 * ElevenLabs access, kept entirely server-side so the API key never reaches
 * the browser.
 *
 * Two jobs: mint short-lived Scribe tokens (the browser opens its own
 * realtime speech-to-text socket with one) and stream text-to-speech audio
 * back for the agent's replies.
 */

import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';

/** Fast, cheap, and low-latency — the replies are one sentence long. */
const TTS_MODEL = 'eleven_flash_v2_5';
const TTS_OUTPUT_FORMAT = 'mp3_44100_128';

let client: ElevenLabsClient | null = null;

function getClient(): ElevenLabsClient {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) throw new Error('ELEVENLABS_API_KEY is not set');
  if (!client) client = new ElevenLabsClient({ apiKey });
  return client;
}

/** A single-use realtime token; expires after 15 minutes and burns on use. */
export async function createScribeToken(): Promise<string> {
  const response = await getClient().tokens.singleUse.create('realtime_scribe');
  return response.token;
}

export async function streamSpeech(text: string): Promise<ReadableStream<Uint8Array>> {
  const voiceId = process.env.ELEVENLABS_VOICE_ID;
  if (!voiceId) throw new Error('ELEVENLABS_VOICE_ID is not set');
  return getClient().textToSpeech.stream(voiceId, {
    text,
    modelId: TTS_MODEL,
    outputFormat: TTS_OUTPUT_FORMAT,
  });
}
