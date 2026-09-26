import { describe, expect, it } from 'vitest';
import {
  POSITION_LIMIT,
  SIZE_MAX,
  describeScene,
  validateToolCall,
} from '../agentTools';
import type { CadCommand, RawToolCall, SceneSummary } from '../agentTools';

/** Validate and unwrap, failing the test if the model call was rejected. */
function accept(name: string, args: Record<string, unknown> = {}): CadCommand {
  const result = validateToolCall({ name, args } satisfies RawToolCall);
  if (!result.ok) throw new Error(`expected "${name}" to validate, got: ${result.error}`);
  return result.command;
}

function reject(name: string, args: Record<string, unknown> = {}): string {
  const result = validateToolCall({ name, args } satisfies RawToolCall);
  if (result.ok) throw new Error(`expected "${name}" to be rejected`);
  return result.error;
}

describe('validateToolCall', () => {
  it('fills in defaults for a bare add_shape', () => {
    expect(accept('add_shape', { shape: 'box' })).toEqual({
      name: 'add_shape',
      shape: 'box',
      width: 1,
      depth: 1,
      height: 1,
      x: 0,
      z: 0,
      color: null,
    });
  });

  it('clamps sizes and positions instead of refusing them', () => {
    const command = accept('add_shape', {
      shape: 'cuboid',
      width: 999,
      depth: -4,
      height: 0,
      x: 50,
      z: -50,
    });
    expect(command).toMatchObject({
      width: SIZE_MAX,
      x: POSITION_LIMIT,
      z: -POSITION_LIMIT,
    });
    if (command.name !== 'add_shape') throw new Error('wrong command');
    expect(command.depth).toBeGreaterThan(0);
    expect(command.height).toBeGreaterThan(0);
  });

  it('keeps round shapes round by tying depth to width', () => {
    const cylinder = accept('add_shape', { shape: 'cylinder', width: 3, depth: 0.5 });
    expect(cylinder).toMatchObject({ width: 3, depth: 3 });
  });

  it('accepts spoken synonyms for the primitives', () => {
    expect(accept('set_shape', { shape: 'cube' })).toEqual({ name: 'set_shape', shape: 'box' });
    expect(accept('set_shape', { shape: 'Ball' })).toEqual({ name: 'set_shape', shape: 'sphere' });
    expect(accept('set_shape', { shape: ' TUBE ' })).toEqual({
      name: 'set_shape',
      shape: 'cylinder',
    });
  });

  it('parses numbers the model sent as strings', () => {
    expect(accept('add_shape', { shape: 'box', height: '2.5' })).toMatchObject({ height: 2.5 });
  });

  it('resolves color names and hex, and refuses anything else', () => {
    expect(accept('set_color', { color: 'red' })).toEqual({
      name: 'set_color',
      color: '#ef4444',
      target: 'last',
    });
    expect(accept('set_color', { color: '#123ABC', target: 'all' })).toEqual({
      name: 'set_color',
      color: '#123abc',
      target: 'all',
    });
    expect(reject('set_color', { color: 'ultraviolet' })).toMatch(/color/i);
    expect(reject('add_shape', { shape: 'box', color: 'not-a-color' })).toMatch(/color/i);
  });

  it('refuses unknown shapes and unknown tools', () => {
    expect(reject('add_shape', { shape: 'dodecahedron' })).toMatch(/shape/i);
    expect(reject('summon_dragon')).toMatch(/unknown tool/i);
  });

  it('passes through the no-argument commands', () => {
    for (const name of [
      'remove_last',
      'clear_scene',
      'describe_scene',
      'export_for_printing',
      'start_camera',
      'stop_camera',
    ]) {
      expect(accept(name)).toEqual({ name });
    }
  });

  it('survives a call with no args at all', () => {
    const result = validateToolCall({ name: 'add_shape' } as RawToolCall);
    expect(result.ok).toBe(false);
  });
});

describe('describeScene', () => {
  const scene = (objects: SceneSummary['objects']): SceneSummary => ({
    objects,
    activeShape: 'box',
    cameraRunning: true,
  });

  it('says so when nothing has been built', () => {
    expect(describeScene(scene([]))).toBe('The scene is empty.');
  });

  it('describes objects in plain language', () => {
    const summary = describeScene(
      scene([
        { id: 1, shape: 'box', width: 2, depth: 1, height: 3, x: 0, z: 0, color: '#fff' },
        { id: 2, shape: 'sphere', width: 1.5, depth: 1.5, height: 1.5, x: 2, z: -1, color: '#fff' },
      ])
    );
    expect(summary).toContain('2 objects');
    expect(summary).toContain('a box 2 by 1 and 3 tall at the center');
    expect(summary).toContain('a sphere 1.5 wide');
    expect(summary).toContain('at 2, -1');
  });
});
