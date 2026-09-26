/**
 * VoiceHud: the visible half of the voice loop.
 *
 * Two pieces, deliberately kept apart:
 *   pill      a small state badge in the camera thumbnail header — the
 *             always-on answer to "is it listening to me right now?"
 *   captions  a glass strip along the bottom of the viewport showing what was
 *             heard and what the agent said back.
 *
 * UI-only module: it mirrors the voice state union rather than importing it,
 * so `src/ui` stays free of `src/voice` (and vice versa). `main.ts` passes the
 * state straight through — the unions are identical on purpose.
 */

/** Mirror of `VoiceState` in `src/voice/VoiceAgent.ts`. */
export type VoiceHudState = 'off' | 'starting' | 'idle' | 'listening' | 'thinking' | 'speaking';

/** Pill copy per state. `off` hides the pill entirely. */
const PILL_LABEL: Record<VoiceHudState, string> = {
  off: '',
  starting: 'Waking up',
  idle: 'Voice on',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Speaking',
};

/** How long a caption stays up after the last update. */
const CAPTION_LINGER_MS = 7_000;

export interface VoiceHudElements {
  /** Badge host, typically in the camera thumbnail header. */
  pill: HTMLElement;
  /** Caption host, typically overlaid on the viewport. */
  captions: HTMLElement;
}

export class VoiceHud {
  private readonly pill: HTMLElement;
  private readonly captions: HTMLElement;
  private readonly heardLine: HTMLElement;
  private readonly replyLine: HTMLElement;
  private hideTimer: number | null = null;

  constructor(elements: VoiceHudElements) {
    this.pill = elements.pill;
    this.captions = elements.captions;

    this.pill.classList.add('voice-pill');
    // Announce state changes without stealing focus from the viewport.
    this.pill.setAttribute('role', 'status');
    this.pill.setAttribute('aria-live', 'polite');

    this.captions.classList.add('voice-captions');
    this.captions.setAttribute('role', 'log');
    this.captions.setAttribute('aria-live', 'polite');

    this.heardLine = document.createElement('p');
    this.heardLine.className = 'voice-caption heard';
    this.replyLine = document.createElement('p');
    this.replyLine.className = 'voice-caption reply';
    this.captions.append(this.heardLine, this.replyLine);

    this.setState('off');
  }

  /** Update the badge. `off` hides it and clears any lingering caption. */
  setState(state: VoiceHudState): void {
    this.pill.dataset.state = state;
    this.pill.textContent = PILL_LABEL[state];
    this.pill.hidden = state === 'off';
    if (state === 'off') this.clear();
  }

  /**
   * Show what the user is saying. `partial` marks an in-flight transcript,
   * which renders dimmer and never starts the auto-hide countdown — the text
   * is still changing.
   */
  showHeard(text: string, { partial = false }: { partial?: boolean } = {}): void {
    this.heardLine.textContent = text ? `“${text}”` : '';
    this.heardLine.classList.toggle('partial', partial);
    this.replyLine.classList.remove('error');
    if (text && !partial) this.reveal();
    else if (text) this.show();
  }

  /** Show the agent's spoken reply. */
  showReply(text: string): void {
    this.replyLine.textContent = text;
    this.replyLine.classList.remove('error');
    this.reveal();
  }

  /** Show a failure in the agent's line, styled as a warning. */
  showError(message: string): void {
    this.replyLine.textContent = message;
    this.replyLine.classList.add('error');
    this.reveal();
  }

  /** Hide the caption strip and drop its contents. */
  clear(): void {
    this.cancelHide();
    this.captions.classList.remove('visible');
    this.heardLine.textContent = '';
    this.replyLine.textContent = '';
  }

  dispose(): void {
    this.cancelHide();
    this.captions.replaceChildren();
    this.captions.classList.remove('voice-captions', 'visible');
    this.captions.removeAttribute('role');
    this.captions.removeAttribute('aria-live');
    this.pill.classList.remove('voice-pill');
    this.pill.removeAttribute('role');
    this.pill.removeAttribute('aria-live');
    this.pill.textContent = '';
    this.pill.hidden = true;
  }

  private show(): void {
    this.cancelHide();
    this.captions.classList.add('visible');
  }

  /** Show, then start the countdown to fade back out. */
  private reveal(): void {
    this.show();
    this.hideTimer = window.setTimeout(() => {
      this.captions.classList.remove('visible');
      this.hideTimer = null;
    }, CAPTION_LINGER_MS);
  }

  private cancelHide(): void {
    if (this.hideTimer === null) return;
    window.clearTimeout(this.hideTimer);
    this.hideTimer = null;
  }
}
