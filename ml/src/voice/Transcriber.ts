/**
 * Transcriber: realtime speech-to-text over ElevenLabs Scribe.
 *
 * The connection holds its own microphone and stays open for the whole
 * session; the TalkGate only mutes and unmutes it. Muting disables the audio
 * track (the browser sends silence), which keeps the socket warm so the first
 * word after a pause is not lost to a reconnect.
 *
 * Commits are manual on purpose. Scribe's own voice-activity detection would
 * decide when a sentence ended from the audio, but we want Presage's view of
 * the user's face to make that call, so `closeMic()` is what seals a turn.
 */

import { CommitStrategy, RealtimeEvents, Scribe } from '@elevenlabs/client';
import type { RealtimeConnection } from '@elevenlabs/client';

/** How often to re-try applying the mute state while the mic track attaches. */
const MUTE_RETRY_MS = 50;
/** Give up (and say so) if muting never takes hold — the mic would be live. */
const MUTE_DEADLINE_MS = 5_000;

/** Words the model should lean towards; these are our whole vocabulary. */
const KEYTERMS = [
  'box',
  'cuboid',
  'cylinder',
  'sphere',
  'taller',
  'wider',
  'undo',
  'clear',
  'export',
];

export interface TranscriberCallbacks {
  /** Live, unstable text while the user is still speaking. */
  onPartial?: (text: string) => void;
  /** A sealed utterance, ready to send to the agent. */
  onCommitted: (text: string) => void;
  onError?: (message: string) => void;
}

export class Transcriber {
  private readonly callbacks: TranscriberCallbacks;
  private connection: RealtimeConnection | null = null;
  private connecting: Promise<void> | null = null;
  private micOpen = false;
  /** Audio has flowed since the last commit — avoids empty-commit throttling. */
  private pendingAudio = false;
  /** A commit we asked for is still in flight. */
  private awaitingCommit = false;
  /** The mute state we want; re-applied until the mic track exists. */
  private desiredMuted = true;
  private muteRetry: number | null = null;
  private muteDeadline = 0;

  constructor(callbacks: TranscriberCallbacks) {
    this.callbacks = callbacks;
  }

  get isConnected(): boolean {
    return this.connection !== null;
  }

  /** Open the socket and grab the microphone (prompts for permission once). */
  async connect(): Promise<void> {
    if (this.connection) return;
    if (this.connecting) return this.connecting;
    this.connecting = this.openConnection().finally(() => {
      this.connecting = null;
    });
    return this.connecting;
  }

  private async openConnection(): Promise<void> {
    const response = await fetch('/api/voice/token', { method: 'POST' });
    if (!response.ok) {
      const detail = (await response.json().catch(() => ({}))) as { error?: string };
      throw new Error(detail.error ?? `Could not get a speech token (${response.status})`);
    }
    const { token } = (await response.json()) as { token: string };

    const connection = Scribe.connect({
      token,
      modelId: 'scribe_v2_realtime',
      commitStrategy: CommitStrategy.MANUAL,
      keyterms: KEYTERMS,
      noVerbatim: true,
      // Presage decides when we are listening, so ignore other voices.
      filterBackgroundAudio: true,
      microphone: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Both handlers check the gate as well as muting. Belt and braces on
    // purpose: if muting ever fails to take hold, the room's conversation must
    // still not reach the screen or the agent.
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (data) => {
      if (!this.micOpen) return;
      if (data.text) this.callbacks.onPartial?.(data.text);
    });
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, (data) => {
      if (!this.awaitingCommit) return;
      this.awaitingCommit = false;
      const text = data.text?.trim();
      if (text) this.callbacks.onCommitted(text);
    });
    connection.on(RealtimeEvents.ERROR, (event) => {
      this.callbacks.onError?.(event.error || 'Speech recognition failed');
    });
    connection.on(RealtimeEvents.CLOSE, () => {
      if (this.connection === connection) {
        this.connection = null;
        this.micOpen = false;
        this.awaitingCommit = false;
        this.clearMuteRetry();
      }
    });

    await new Promise<void>((resolve, reject) => {
      connection.on(RealtimeEvents.OPEN, () => resolve());
      connection.on(RealtimeEvents.AUTH_ERROR, (event) =>
        reject(new Error(event.error || 'Speech token was rejected'))
      );
    });

    this.connection = connection;
    // Start silent: nothing is transcribed until the gate opens.
    this.setMuted(true);
  }

  /** Start capturing. Reconnects transparently if the socket timed out. */
  async openMic(): Promise<void> {
    if (!this.connection) await this.connect();
    this.micOpen = true;
    this.pendingAudio = true;
    this.setMuted(false);
  }

  /** Stop capturing and seal the utterance. */
  closeMic(): void {
    if (!this.connection || !this.micOpen) return;
    this.micOpen = false;
    this.setMuted(true);
    if (!this.pendingAudio) return;
    this.pendingAudio = false;
    try {
      this.awaitingCommit = true;
      this.connection.commit();
    } catch (error) {
      this.awaitingCommit = false;
      this.callbacks.onError?.(error instanceof Error ? error.message : String(error));
    }
  }

  private setMuted(muted: boolean): void {
    this.desiredMuted = muted;
    this.muteDeadline = Date.now() + MUTE_DEADLINE_MS;
    this.applyMute();
  }

  /**
   * Apply the wanted mute state, retrying until it sticks.
   *
   * Scribe attaches the microphone `MediaStreamTrack` asynchronously, a moment
   * after the socket opens, and `mute()` throws until it exists. Swallowing
   * that failure leaves the microphone live: Scribe then transcribes the whole
   * room while the gate is shut, and nothing ever gets committed. So keep
   * trying, and complain if it never works.
   */
  private applyMute(): void {
    this.clearMuteRetry();
    const connection = this.connection;
    if (!connection) return;
    try {
      if (this.desiredMuted) connection.mute();
      else connection.unmute();
      return;
    } catch (error) {
      if (Date.now() >= this.muteDeadline) {
        this.callbacks.onError?.(
          `Could not ${this.desiredMuted ? 'mute' : 'unmute'} the microphone: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
        return;
      }
    }
    this.muteRetry = window.setTimeout(() => {
      this.muteRetry = null;
      this.applyMute();
    }, MUTE_RETRY_MS);
  }

  private clearMuteRetry(): void {
    if (this.muteRetry === null) return;
    window.clearTimeout(this.muteRetry);
    this.muteRetry = null;
  }

  close(): void {
    const connection = this.connection;
    this.connection = null;
    this.micOpen = false;
    this.pendingAudio = false;
    this.awaitingCommit = false;
    this.clearMuteRetry();
    connection?.close();
  }
}
