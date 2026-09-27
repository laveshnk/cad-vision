/**
 * Presage SmartSpectra talk detection.
 *
 * SmartSpectra has no browser SDK, so the browser ships small JPEG frames from
 * the webcam it is already running and this module feeds them to the Node SDK
 * in custom-input mode. The only metric we ask for is the face bundle, and the
 * only field we use is `face.talking`.
 *
 * Custom input stays in `kStarting` until the first frame arrives, and a frame
 * sent after the pipeline has fallen into `kError` throws "not in a valid
 * state" on every subsequent call. So frames are accepted while starting or
 * running, and the first SDK error stops the session for good.
 *
 * The native SDK keeps one pipeline per process. The next session waits until
 * the previous `destroy()` has finished.
 */

import jpeg from 'jpeg-js';
import type { PixelFormatValue, ProcessingStatusValue, SmartSpectraSDK } from '@smartspectra/node-sdk';

export type TalkingListener = (talking: boolean) => void;
/** The pipeline died. The caller should stop sending frames and fall back. */
export type FatalListener = (reason: string) => void;

export interface PresageSession {
  /** Decode and forward one webcam frame. Ignored once closed or failed. */
  pushFrame(jpegFrame: Buffer): void;
  close(): Promise<void>;
}

/**
 * The SDK requires strictly increasing capture timestamps in microseconds and
 * rejects wall-clock values, so this tracks a monotonic clock per session.
 */
function createClock(): () => number {
  const startedAt = process.hrtime.bigint();
  let previous = 0;
  return () => {
    const elapsed = Number((process.hrtime.bigint() - startedAt) / 1000n);
    previous = Math.max(elapsed, previous + 1);
    return previous;
  };
}

/** Holds until the session that acquired it has finished `destroy()`. */
let sessionGate: Promise<void> = Promise.resolve();

export async function createPresageSession(
  onTalking: TalkingListener,
  onFatal?: FatalListener,
): Promise<PresageSession> {
  const apiKey = process.env.PRESAGE_API_KEY;
  if (!apiKey) throw new Error('PRESAGE_API_KEY is not set');

  const previous = sessionGate;
  let releaseGate: () => void = () => {};
  sessionGate = new Promise<void>((resolve) => {
    releaseGate = resolve;
  });
  await previous;

  try {
    return await openSession(apiKey, onTalking, onFatal, releaseGate);
  } catch (error) {
    releaseGate();
    throw error;
  }
}

async function openSession(
  apiKey: string,
  onTalking: TalkingListener,
  onFatal: FatalListener | undefined,
  releaseGate: () => void,
): Promise<PresageSession> {
  const { SmartSpectraSDK, PixelFormat, FrameTransform, ProcessingStatus, faceMetrics } =
    await import('@smartspectra/node-sdk');
  const { decodeMetrics } = await import('@smartspectra/node-sdk/messages');

  const sdk: SmartSpectraSDK = new SmartSpectraSDK({
    apiKey,
    requestedMetrics: [...faceMetrics],
  });

  let closed = false;
  let failed = false;
  let gateReleased = false;

  const finishGate = () => {
    if (gateReleased) return;
    gateReleased = true;
    releaseGate();
  };

  /** One report, then silence. Further frames repeat the same failure. */
  const fail = (reason: string) => {
    if (closed || failed) return;
    failed = true;
    console.error(`[presage] ${reason}`);
    onFatal?.(reason);
  };

  sdk.on('processingStatus', (status: ProcessingStatusValue) => {
    if (status === ProcessingStatus.kError) fail('Presage entered an error state');
  });
  sdk.on('metrics', (buffer) => {
    const metrics = decodeMetrics(buffer);
    const latest = metrics.face?.talking?.at(-1);
    if (latest) onTalking(latest.detected === true);
  });
  sdk.on('error', (code, message) => {
    fail(presageFailureReason(code, message));
  });

  sdk.useCustomInput(FrameTransform.kNone);
  sdk.start();

  const nextTimestamp = createClock();
  const rgbFormat: PixelFormatValue = PixelFormat.kRGB;
  const accepting = new Set<ProcessingStatusValue>([
    ProcessingStatus.kStarting,
    ProcessingStatus.kRunning,
  ]);

  return {
    pushFrame(jpegFrame: Buffer): void {
      if (closed || failed || !accepting.has(sdk.processingStatus)) return;
      try {
        const { width, height, data } = jpeg.decode(jpegFrame, { formatAsRGBA: false });
        if (width < 2 || height < 2 || data.length < width * height * 3) return;
        sdk.sendFrame(data, width, height, width * 3, rgbFormat, nextTimestamp());
      } catch (error) {
        fail(error instanceof Error ? error.message : String(error));
      }
    },
    async close(): Promise<void> {
      if (closed) return;
      closed = true;
      try {
        await sdk.stopAsync();
        await sdk.destroy();
      } catch (error) {
        console.error('[presage] shutdown failed:', error);
      } finally {
        finishGate();
      }
    },
  };
}

/**
 * The SDK reports a dead pipeline as "processing failed", and the frame calls
 * after that throw "not in a valid state". Both are the same outcome when
 * device pairing was refused: the Physiology API key was rejected.
 */
function presageFailureReason(code: number, message: string): string {
  const text = message.trim();
  if (code === 2 || /auth/i.test(text)) {
    return 'Presage rejected the API key. Check PRESAGE_API_KEY from physiology.presagetech.com.';
  }
  if (code === 8) {
    return 'Presage could not start measuring. The API key was likely rejected (401); check PRESAGE_API_KEY.';
  }
  return text ? `sdk error ${code}: ${text}` : `sdk error ${code}`;
}
