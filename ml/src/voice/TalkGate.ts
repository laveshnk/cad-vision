/**
 * TalkGate: decides when the microphone should be listening, from a stream of
 * "is the face talking" observations.
 *
 * Presage reports mouth movement per frame, which is too twitchy to wire
 * straight to a microphone — a single dropped frame mid-sentence would cut the
 * user off. This applies hysteresis: talking has to persist briefly before the
 * gate opens, and silence has to persist longer before it closes, so natural
 * pauses inside a sentence do not end the turn.
 *
 * While the agent is speaking the gate stays shut and sustained talking raises
 * `barge_in` instead, letting the caller stop playback first. Because the
 * signal is the user's face rather than the microphone, the agent's own voice
 * coming out of the speakers can never trigger this.
 *
 * Pure logic: time arrives as an argument, nothing is scheduled here, so the
 * whole thing is unit-testable with a synthetic clock.
 */

export interface TalkGateOptions {
  /** Sustained talking before the mic opens. */
  openDelayMs?: number;
  /** Sustained silence before the turn is considered finished. */
  closeDelayMs?: number;
  /** Sustained talking over the agent's reply before `barge_in` fires. */
  bargeInDelayMs?: number;
  /**
   * Hard cap on one open turn. A detector that gets stuck reporting "talking"
   * would otherwise hold the microphone open forever and never seal anything,
   * so the gate closes itself and waits for real silence before reopening.
   * Set to 0 to disable.
   */
  maxOpenMs?: number;
}

export type TalkGateEvent = 'open' | 'close' | 'barge_in';

export class TalkGate {
  private readonly openDelayMs: number;
  private readonly closeDelayMs: number;
  private readonly bargeInDelayMs: number;
  private readonly maxOpenMs: number;

  private open = false;
  private speaking = false;
  private talking = false;
  /** When the current talking/silent run began. */
  private since = 0;
  /** When the gate last opened, for the hard cap. */
  private openedAt = 0;
  /** Force-closed on the cap: stay shut until silence proves the signal moves. */
  private latched = false;
  private bargeInSent = false;

  constructor(options: TalkGateOptions = {}) {
    this.openDelayMs = options.openDelayMs ?? 150;
    this.closeDelayMs = options.closeDelayMs ?? 700;
    this.bargeInDelayMs = options.bargeInDelayMs ?? 300;
    this.maxOpenMs = options.maxOpenMs ?? 20_000;
  }

  get isOpen(): boolean {
    return this.open;
  }

  /** Tell the gate whether the agent is currently talking back. */
  setSpeaking(speaking: boolean): void {
    if (this.speaking === speaking) return;
    this.speaking = speaking;
    this.bargeInSent = false;
  }

  /**
   * Feed one observation. Safe to call on every Presage update and on a timer
   * in between — the delays are measured from when the run started, not from
   * the previous call, so a sparse stream of edges behaves the same as a dense
   * one.
   */
  observe(talking: boolean, now: number): TalkGateEvent[] {
    if (talking !== this.talking) {
      this.talking = talking;
      this.since = now;
      if (!talking) {
        this.bargeInSent = false;
        this.latched = false;
      }
    }
    const heldMs = now - this.since;
    const events: TalkGateEvent[] = [];

    if (this.speaking) {
      // Interrupting: don't open the mic on top of our own playback, just
      // report it so the caller can stop speaking first.
      if (talking && !this.bargeInSent && heldMs >= this.bargeInDelayMs) {
        this.bargeInSent = true;
        events.push('barge_in');
      }
      return events;
    }

    if (this.open) {
      if (this.maxOpenMs > 0 && now - this.openedAt >= this.maxOpenMs) {
        this.open = false;
        // Only latch if the signal is still stuck on; a normal long sentence
        // that has already stopped should be free to start again immediately.
        this.latched = talking;
        events.push('close');
      } else if (!talking && heldMs >= this.closeDelayMs) {
        this.open = false;
        events.push('close');
      }
    } else if (talking && !this.latched && heldMs >= this.openDelayMs) {
      this.open = true;
      this.openedAt = now;
      events.push('open');
    }
    return events;
  }

  /** Forget all state (voice turned off, or the signal was lost). */
  reset(): void {
    this.open = false;
    this.speaking = false;
    this.talking = false;
    this.since = 0;
    this.openedAt = 0;
    this.latched = false;
    this.bargeInSent = false;
  }
}
