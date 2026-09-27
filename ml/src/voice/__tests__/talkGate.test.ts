import { describe, expect, it } from 'vitest';
import { TalkGate } from '../TalkGate';

const OPTIONS = { openDelayMs: 150, closeDelayMs: 700, bargeInDelayMs: 300 };

describe('TalkGate', () => {
  it('waits for sustained talking before opening the mic', () => {
    const gate = new TalkGate(OPTIONS);
    expect(gate.observe(true, 1000)).toEqual([]);
    expect(gate.observe(true, 1100)).toEqual([]);
    expect(gate.isOpen).toBe(false);
    expect(gate.observe(true, 1150)).toEqual(['open']);
    expect(gate.isOpen).toBe(true);
  });

  it('opens only once while talking continues', () => {
    const gate = new TalkGate(OPTIONS);
    gate.observe(true, 0);
    expect(gate.observe(true, 200)).toEqual(['open']);
    expect(gate.observe(true, 400)).toEqual([]);
  });

  it('holds the mic open through a pause inside a sentence', () => {
    const gate = new TalkGate(OPTIONS);
    gate.observe(true, 0);
    gate.observe(true, 200);

    gate.observe(false, 300);
    expect(gate.observe(false, 900)).toEqual([]); // 600ms of silence, under the close delay
    gate.observe(true, 950);
    expect(gate.isOpen).toBe(true);
  });

  it('closes after silence outlasts the close delay', () => {
    const gate = new TalkGate(OPTIONS);
    gate.observe(true, 0);
    gate.observe(true, 200);

    gate.observe(false, 1000);
    expect(gate.observe(false, 1699)).toEqual([]);
    expect(gate.observe(false, 1700)).toEqual(['close']);
    expect(gate.isOpen).toBe(false);
  });

  it('reports a barge-in instead of opening while the agent speaks', () => {
    const gate = new TalkGate(OPTIONS);
    gate.setSpeaking(true);

    expect(gate.observe(true, 1000)).toEqual([]);
    expect(gate.observe(true, 1299)).toEqual([]);
    expect(gate.observe(true, 1300)).toEqual(['barge_in']);
    expect(gate.isOpen).toBe(false);
  });

  it('raises at most one barge-in per interruption', () => {
    const gate = new TalkGate(OPTIONS);
    gate.setSpeaking(true);
    gate.observe(true, 0);
    expect(gate.observe(true, 300)).toEqual(['barge_in']);
    expect(gate.observe(true, 900)).toEqual([]);

    // A fresh interruption after a silent stretch is allowed to fire again.
    gate.observe(false, 1000);
    gate.observe(true, 1100);
    expect(gate.observe(true, 1400)).toEqual(['barge_in']);
  });

  it('opens immediately once playback stops mid-interruption', () => {
    const gate = new TalkGate(OPTIONS);
    gate.setSpeaking(true);
    gate.observe(true, 1000);
    expect(gate.observe(true, 1300)).toEqual(['barge_in']);

    gate.setSpeaking(false);
    // The talking run started at 1000, so the open delay has long since passed.
    expect(gate.observe(true, 1310)).toEqual(['open']);
  });

  it('treats a sparse stream of edges like a dense one', () => {
    const dense = new TalkGate(OPTIONS);
    let denseOpened = false;
    for (let t = 0; t <= 200; t += 20) {
      if (dense.observe(true, t).includes('open')) denseOpened = true;
    }

    // Presage only reports changes, so the gate must measure from the start of
    // the talking run rather than from the previous call.
    const sparse = new TalkGate(OPTIONS);
    sparse.observe(true, 0);

    expect(denseOpened).toBe(true);
    expect(sparse.observe(true, 200)).toEqual(['open']);
    expect(sparse.isOpen).toBe(dense.isOpen);
  });

  it('seals a turn that has run past the hard cap', () => {
    const gate = new TalkGate({ ...OPTIONS, maxOpenMs: 5_000 });
    gate.observe(true, 0);
    expect(gate.observe(true, 200)).toEqual(['open']);
    expect(gate.observe(true, 4_000)).toEqual([]);
    expect(gate.observe(true, 5_200)).toEqual(['close']);
    expect(gate.isOpen).toBe(false);
  });

  it('will not reopen on a signal stuck at "talking" until real silence', () => {
    const gate = new TalkGate({ ...OPTIONS, maxOpenMs: 5_000 });
    gate.observe(true, 0);
    gate.observe(true, 200);
    gate.observe(true, 5_200); // forced close

    // Still stuck on: the gate must not immediately reopen and loop forever.
    expect(gate.observe(true, 6_000)).toEqual([]);
    expect(gate.observe(true, 20_000)).toEqual([]);

    // Silence proves the signal is alive again, so the next run may open.
    gate.observe(false, 20_100);
    expect(gate.observe(true, 20_200)).toEqual([]);
    expect(gate.observe(true, 20_400)).toEqual(['open']);
  });

  it('lets a long but finished sentence start a new turn right away', () => {
    const gate = new TalkGate({ ...OPTIONS, maxOpenMs: 5_000 });
    gate.observe(true, 0);
    gate.observe(true, 200);
    // Silence arrives first, then the cap trips on the same run.
    gate.observe(false, 4_900);
    expect(gate.observe(false, 5_200)).toEqual(['close']);
    expect(gate.observe(true, 5_300)).toEqual([]);
    expect(gate.observe(true, 5_500)).toEqual(['open']);
  });

  it('forgets everything on reset', () => {
    const gate = new TalkGate(OPTIONS);
    gate.observe(true, 0);
    gate.observe(true, 200);
    expect(gate.isOpen).toBe(true);

    gate.reset();
    expect(gate.isOpen).toBe(false);
    expect(gate.observe(true, 300)).toEqual([]);
    expect(gate.observe(true, 450)).toEqual(['open']);
  });
});
