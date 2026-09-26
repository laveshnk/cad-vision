/**
 * PresageFrames: streams webcam stills to the voice backend and reports back
 * whether the user's mouth is moving.
 *
 * Presage SmartSpectra has no browser build, so the analysis runs server-side.
 * This piggybacks on the `<video>` element MediaPipe is already using — no
 * second camera, no second `getUserMedia` prompt — and sends a small JPEG a
 * few times a second over a WebSocket.
 *
 * Everything here is best-effort. If the socket cannot be opened, or the
 * server has no Presage key, `onStatus` reports `unavailable` and the caller
 * falls back to a manual voice toggle.
 */

export type PresageStatus = 'idle' | 'connecting' | 'ready' | 'unavailable';

export interface PresageFramesOptions {
  /** Frames per second sent to the server. Talk detection needs very little. */
  fps?: number;
  /** Long edge of the JPEG sent upstream. */
  frameWidth?: number;
  jpegQuality?: number;
  /** Delay before re-opening a dropped socket. */
  reconnectDelayMs?: number;
}

export interface PresageFramesCallbacks {
  onTalking: (talking: boolean) => void;
  onStatus?: (status: PresageStatus, reason?: string) => void;
}

interface ServerMessage {
  type: 'ready' | 'talking' | 'unavailable';
  talking?: boolean;
  reason?: string;
}

export class PresageFrames {
  private readonly options: Required<PresageFramesOptions>;
  private readonly callbacks: PresageFramesCallbacks;
  private readonly canvas = document.createElement('canvas');

  private socket: WebSocket | null = null;
  private video: HTMLVideoElement | null = null;
  private timer: number | null = null;
  private reconnectTimer: number | null = null;
  private status: PresageStatus = 'idle';
  /** Set once the server says it cannot run Presage at all — stop retrying. */
  private disabled = false;
  private sending = false;

  constructor(callbacks: PresageFramesCallbacks, options: PresageFramesOptions = {}) {
    this.callbacks = callbacks;
    this.options = {
      fps: options.fps ?? 10,
      frameWidth: options.frameWidth ?? 480,
      jpegQuality: options.jpegQuality ?? 0.7,
      reconnectDelayMs: options.reconnectDelayMs ?? 3_000,
    };
  }

  get currentStatus(): PresageStatus {
    return this.status;
  }

  /** Begin streaming frames from a playing video element. */
  start(video: HTMLVideoElement): void {
    if (this.disabled) return;
    this.video = video;
    this.connect();
  }

  /**
   * Report that mouth detection cannot run right now — typically because the
   * camera is off. Unlike a server-side failure this is not permanent, so
   * `start()` can still bring it up later.
   */
  markUnavailable(reason: string): void {
    this.setStatus('unavailable', reason);
  }

  stop(): void {
    this.clearTimers();
    this.video = null;
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    this.setStatus('idle');
  }

  private setStatus(status: PresageStatus, reason?: string): void {
    if (this.status === status) return;
    this.status = status;
    this.callbacks.onStatus?.(status, reason);
  }

  private connect(): void {
    if (this.disabled || this.socket) return;
    this.setStatus('connecting');
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${scheme}://${window.location.host}/ws/presage`);
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.addEventListener('message', (event) => {
      if (typeof event.data !== 'string') return;
      let message: ServerMessage;
      try {
        message = JSON.parse(event.data) as ServerMessage;
      } catch {
        return;
      }
      if (message.type === 'ready') {
        this.setStatus('ready');
        this.startFrameLoop();
      } else if (message.type === 'talking') {
        this.callbacks.onTalking(message.talking === true);
      } else if (message.type === 'unavailable') {
        // The server can't do this at all; retrying would just spin.
        this.disabled = true;
        this.setStatus('unavailable', message.reason);
      }
    });

    socket.addEventListener('close', () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.clearTimers();
      if (this.disabled) return;
      this.setStatus('connecting');
      this.scheduleReconnect();
    });

    socket.addEventListener('error', () => socket.close());
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null || !this.video) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (this.video) this.connect();
    }, this.options.reconnectDelayMs);
  }

  private startFrameLoop(): void {
    if (this.timer !== null) return;
    this.timer = window.setInterval(() => {
      void this.sendFrame();
    }, Math.round(1000 / this.options.fps));
  }

  private clearTimers(): void {
    if (this.timer !== null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private async sendFrame(): Promise<void> {
    const video = this.video;
    const socket = this.socket;
    // JPEG encoding is async; skip a tick rather than queue frames up.
    if (this.sending || !video || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    this.sending = true;
    try {
      const scale = this.options.frameWidth / video.videoWidth;
      this.canvas.width = this.options.frameWidth;
      this.canvas.height = Math.round(video.videoHeight * scale);
      const context = this.canvas.getContext('2d');
      if (!context) return;
      context.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        this.canvas.toBlob(resolve, 'image/jpeg', this.options.jpegQuality)
      );
      if (!blob || this.socket !== socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(await blob.arrayBuffer());
    } catch (error) {
      console.error('[voice] failed to send a frame to Presage:', error);
    } finally {
      this.sending = false;
    }
  }
}
