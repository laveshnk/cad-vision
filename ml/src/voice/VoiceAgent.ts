/**
 * VoiceAgent: the facade that turns "user said something" into "the scene
 * changed and the agent said something back".
 *
 * Wiring, in order:
 *
 *   PresageFrames  webcam stills -> the server -> is the mouth moving
 *     -> TalkGate     hysteresis; opens and closes the microphone
 *     -> Transcriber  ElevenLabs Scribe; emits a sealed utterance
 *     -> /api/agent/turn  Gemini; replies with words or tool calls
 *     -> ToolExecutor tool calls run against the CAD scene
 *     -> Speaker      the reply is spoken, interruptible
 *
 * The agent never imports the CAD, UI or vision layers. It reaches the scene
 * only through the injected `ToolExecutor` and `getScene` callbacks, which is
 * what keeps `src/voice` swappable and `main.ts` the only bridge.
 */

import { PresageFrames } from './PresageFrames';
import type { PresageFramesOptions, PresageStatus } from './PresageFrames';
import { Speaker } from './Speaker';
import { TalkGate } from './TalkGate';
import type { TalkGateOptions } from './TalkGate';
import { Transcriber } from './Transcriber';
import { validateToolCall } from '../../shared/agentTools';
import type { RawToolCall, SceneSummary, ToolExecutor, ToolOutcome } from '../../shared/agentTools';

/** How often the gate is re-evaluated; the server only sends state changes. */
const GATE_TICK_MS = 100;
/** Cap on tool-call rounds per utterance, so a confused model cannot loop. */
const MAX_TOOL_ROUNDS = 3;

export type VoiceState = 'off' | 'starting' | 'idle' | 'listening' | 'thinking' | 'speaking';

export interface VoiceAgentOptions {
  /** Runs validated commands against the scene. */
  executor: ToolExecutor;
  /** Current scene, read fresh at the start of every round. */
  getScene: () => SceneSummary;
  /** Base64 JPEG of the viewport so the model can see the build. */
  getSnapshot?: () => string | null;
  talkGate?: TalkGateOptions;
  presage?: PresageFramesOptions;
}

export interface VoiceAgentCallbacks {
  onState?: (state: VoiceState) => void;
  /** Live transcript while the user is speaking. */
  onPartial?: (text: string) => void;
  /** The sealed utterance that was sent to the agent. */
  onTranscript?: (text: string) => void;
  onReply?: (text: string) => void;
  onError?: (message: string) => void;
  /** Fires when hands-free detection turns out to be unavailable. */
  onPresageStatus?: (status: PresageStatus, reason?: string) => void;
}

export class VoiceAgent {
  private readonly options: VoiceAgentOptions;
  private readonly callbacks: VoiceAgentCallbacks;
  private readonly gate: TalkGate;
  private readonly presage: PresageFrames;
  private readonly transcriber: Transcriber;
  private readonly speaker: Speaker;

  private readonly sessionId = crypto.randomUUID();
  private state: VoiceState = 'off';
  private running = false;
  private lastTalking = false;
  private tickTimer: number | null = null;
  private manualListening = false;
  private busy = false;

  constructor(options: VoiceAgentOptions, callbacks: VoiceAgentCallbacks = {}) {
    this.options = options;
    this.callbacks = callbacks;
    this.gate = new TalkGate(options.talkGate);

    this.presage = new PresageFrames(
      {
        onTalking: (talking) => {
          this.lastTalking = talking;
          this.pumpGate();
        },
        onStatus: (status, reason) => this.callbacks.onPresageStatus?.(status, reason),
      },
      options.presage
    );

    this.transcriber = new Transcriber({
      onPartial: (text) => this.callbacks.onPartial?.(text),
      onCommitted: (text) => void this.handleUtterance(text),
      onError: (message) => this.callbacks.onError?.(message),
    });

    this.speaker = new Speaker({
      onStart: () => {
        this.gate.setSpeaking(true);
        this.setState('speaking');
      },
      onEnd: () => {
        this.gate.setSpeaking(false);
        if (this.running && !this.busy) this.setState(this.gate.isOpen ? 'listening' : 'idle');
      },
      onError: (message) => this.callbacks.onError?.(message),
    });
  }

  get currentState(): VoiceState {
    return this.state;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /**
   * True whenever hands-free detection is not actually running, meaning the
   * caller must expose a manual listen button or the microphone will never
   * open. Deliberately covers `connecting` and `idle` as well as outright
   * failure — "not ready yet" leaves the user just as stuck as "never".
   */
  get needsManualTrigger(): boolean {
    return this.presage.currentStatus !== 'ready';
  }

  /**
   * Start listening. `video` is the element the webcam is already playing
   * into; pass `null` to run without hands-free detection (manual only).
   */
  async start(video: HTMLVideoElement | null): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.setState('starting');
    try {
      await this.transcriber.connect();
    } catch (error) {
      this.running = false;
      this.setState('off');
      this.callbacks.onError?.(error instanceof Error ? error.message : String(error));
      return;
    }
    // Mouth detection watches the tracker's own video element, so without a
    // camera there is nothing to watch. Say so loudly: the caller has to offer
    // a manual trigger, or the microphone would simply never open.
    if (video) this.presage.start(video);
    else this.presage.markUnavailable('The camera is off, so mouth detection cannot run.');
    this.tickTimer = window.setInterval(() => this.pumpGate(), GATE_TICK_MS);
    this.setState('idle');
  }

  /**
   * Hand over the camera once it starts, so hands-free detection can take over
   * from the manual button mid-session.
   */
  attachVideo(video: HTMLVideoElement): void {
    if (!this.running) return;
    this.presage.start(video);
  }

  /** The camera went away; fall back to the manual trigger. */
  detachVideo(): void {
    if (!this.running) return;
    this.presage.stop();
    this.manualListening = false;
    this.lastTalking = false;
    this.presage.markUnavailable('The camera is off, so mouth detection cannot run.');
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    if (this.tickTimer !== null) {
      window.clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
    this.presage.stop();
    this.transcriber.close();
    this.speaker.stop();
    this.gate.reset();
    this.lastTalking = false;
    this.manualListening = false;
    this.setState('off');
    void fetch('/api/agent/end', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ sessionId: this.sessionId }),
    }).catch(() => {
      /* the session expires on its own */
    });
  }

  /**
   * Manual push-to-talk, for when Presage is unavailable. Drives the same
   * gate, so the rest of the pipeline cannot tell the difference.
   */
  toggleListening(): void {
    if (!this.running) return;
    this.manualListening = !this.manualListening;
    this.lastTalking = this.manualListening;
    this.pumpGate();
  }

  /**
   * Speak a line directly, without a model round-trip. Used for greetings and
   * other app-authored asides; it still routes through the Speaker, so the
   * gate suppresses self-triggering and barge-in still interrupts it.
   */
  async say(text: string): Promise<void> {
    this.callbacks.onReply?.(text);
    await this.speaker.speak(text);
  }

  private pumpGate(): void {
    if (!this.running) return;
    const talking = this.presage.currentStatus === 'ready' ? this.lastTalking : this.manualListening;
    for (const event of this.gate.observe(talking, performance.now())) {
      if (event === 'barge_in') {
        this.speaker.stop();
      } else if (event === 'open') {
        void this.transcriber.openMic().catch((error: unknown) => {
          this.callbacks.onError?.(error instanceof Error ? error.message : String(error));
        });
        this.setState('listening');
      } else if (event === 'close') {
        this.transcriber.closeMic();
        if (!this.busy) this.setState('idle');
      }
    }
  }

  private setState(state: VoiceState): void {
    if (this.state === state) return;
    this.state = state;
    this.callbacks.onState?.(state);
  }

  /* ---- Turn loop ---- */

  private async handleUtterance(text: string): Promise<void> {
    if (!this.running || this.busy) return;
    this.busy = true;
    this.callbacks.onTranscript?.(text);
    this.setState('thinking');
    try {
      let response = await this.postTurn({
        text,
        scene: this.options.getScene(),
        snapshotJpegBase64: this.options.getSnapshot?.() ?? undefined,
      });

      for (let round = 0; round < MAX_TOOL_ROUNDS && response.toolCalls.length > 0; round += 1) {
        const outcomes = await this.runToolCalls(response.toolCalls);
        response = await this.postTurn({ toolOutcomes: outcomes, scene: this.options.getScene() });
      }

      if (response.text) {
        this.callbacks.onReply?.(response.text);
        await this.speaker.speak(response.text);
      }
    } catch (error) {
      this.callbacks.onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      this.busy = false;
      if (this.running && !this.speaker.isSpeaking) {
        this.setState(this.gate.isOpen ? 'listening' : 'idle');
      }
    }
  }

  /** Validate then run each call, reporting failures back to the model so it
   *  can correct itself out loud instead of silently doing nothing. */
  private async runToolCalls(calls: RawToolCall[]): Promise<ToolOutcome[]> {
    const outcomes: ToolOutcome[] = [];
    for (const call of calls) {
      const validation = validateToolCall(call);
      if (!validation.ok) {
        outcomes.push({ id: call.id, name: call.name, ok: false, detail: validation.error });
        continue;
      }
      try {
        const detail = await this.options.executor.execute(validation.command);
        outcomes.push({ id: call.id, name: call.name, ok: true, detail });
      } catch (error) {
        outcomes.push({
          id: call.id,
          name: call.name,
          ok: false,
          detail: error instanceof Error ? error.message : String(error),
        });
      }
    }
    return outcomes;
  }

  private async postTurn(body: {
    text?: string;
    toolOutcomes?: ToolOutcome[];
    scene: SceneSummary;
    snapshotJpegBase64?: string;
  }): Promise<{ text: string; toolCalls: RawToolCall[] }> {
    const response = await fetch('/api/agent/turn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ sessionId: this.sessionId, ...body }),
    });
    if (!response.ok) {
      const detail = (await response.json().catch(() => ({}))) as { error?: string };
      throw new Error(detail.error ?? `The agent is unreachable (${response.status})`);
    }
    return (await response.json()) as { text: string; toolCalls: RawToolCall[] };
  }

  dispose(): void {
    this.stop();
    this.speaker.dispose();
  }
}
