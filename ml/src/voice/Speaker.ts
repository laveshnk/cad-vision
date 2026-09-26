/**
 * Speaker: plays the agent's replies through ElevenLabs text-to-speech.
 *
 * Replies are one short sentence, so this fetches the whole clip and plays it
 * rather than streaming chunks through MediaSource — a fraction of the code
 * for a barely different time-to-first-sound at this length.
 *
 * `stop()` is what makes interruption work: the TalkGate raises `barge_in`
 * when the user starts talking over a reply, and the audio cuts immediately.
 */

export interface SpeakerCallbacks {
  onStart?: () => void;
  onEnd?: () => void;
  onError?: (message: string) => void;
}

export class Speaker {
  private readonly callbacks: SpeakerCallbacks;
  private readonly audio = new Audio();
  private objectUrl: string | null = null;
  private pending: AbortController | null = null;
  private speaking = false;

  constructor(callbacks: SpeakerCallbacks = {}) {
    this.callbacks = callbacks;
    this.audio.addEventListener('ended', () => this.finish());
    this.audio.addEventListener('error', () => this.finish());
  }

  get isSpeaking(): boolean {
    return this.speaking;
  }

  /** Speak one line. Resolves when playback finishes or is interrupted. */
  async speak(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) return;
    this.stop();

    const controller = new AbortController();
    this.pending = controller;
    try {
      const response = await fetch('/api/voice/tts', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = (await response.json().catch(() => ({}))) as { error?: string };
        throw new Error(detail.error ?? `Speech failed (${response.status})`);
      }
      const blob = await response.blob();
      if (controller.signal.aborted) return;

      this.releaseUrl();
      this.objectUrl = URL.createObjectURL(blob);
      this.audio.src = this.objectUrl;
      this.speaking = true;
      this.callbacks.onStart?.();

      await new Promise<void>((resolve) => {
        const done = () => {
          this.audio.removeEventListener('ended', done);
          this.audio.removeEventListener('error', done);
          resolve();
        };
        this.audio.addEventListener('ended', done);
        this.audio.addEventListener('error', done);
        void this.audio.play().catch(done);
      });
    } catch (error) {
      if (controller.signal.aborted) return;
      this.callbacks.onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      if (this.pending === controller) this.pending = null;
      this.finish();
    }
  }

  /** Cut playback short (used for barge-in). */
  stop(): void {
    this.pending?.abort();
    this.pending = null;
    if (this.audio.src) {
      if (!this.audio.paused) this.audio.pause();
      this.audio.currentTime = 0;
    }
    this.finish();
  }

  private finish(): void {
    if (!this.speaking) return;
    this.speaking = false;
    this.callbacks.onEnd?.();
  }

  private releaseUrl(): void {
    if (!this.objectUrl) return;
    URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
  }

  dispose(): void {
    this.stop();
    this.releaseUrl();
    this.audio.removeAttribute('src');
  }
}
