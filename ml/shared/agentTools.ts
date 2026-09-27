/**
 * Shared voice-agent contract: the tool vocabulary the model may call, the
 * scene summary it reasons over, and validators that clamp model output into
 * ranges the CAD layer can survive.
 *
 * Imported by both the browser (`src/voice`) and the Node server
 * (`server/agent.ts`), so this module stays dependency-free — no DOM, no Node
 * built-ins, no SDK types. The server hands the declarations to Gemini; the
 * browser validates whatever comes back before touching the scene, so a
 * hallucinated argument can never reach `CadBuilder`.
 */

export type ShapeId = 'box' | 'cuboid' | 'cylinder' | 'sphere';

export const SHAPE_IDS: readonly ShapeId[] = ['box', 'cuboid', 'cylinder', 'sphere'];

/** Size limits in world units — small enough to stay inside the grid. */
export const SIZE_MIN = 0.1;
export const SIZE_MAX = 8;
/** The camera focus is locked to the origin, so keep placements near it. */
export const POSITION_LIMIT = 6;

/** Colors the model can name instead of guessing hex. */
export const NAMED_COLORS: Readonly<Record<string, string>> = {
  red: '#ef4444',
  orange: '#f97316',
  yellow: '#eab308',
  green: '#22c55e',
  teal: '#14b8a6',
  blue: '#3b82f6',
  purple: '#a855f7',
  pink: '#ec4899',
  white: '#e5e7eb',
  gray: '#6b7280',
  grey: '#6b7280',
  black: '#27272a',
};

/* -------------------------------------------------------------------------- */
/* Scene summary (browser -> model)                                            */
/* -------------------------------------------------------------------------- */

export interface SceneObjectSummary {
  id: number;
  shape: ShapeId;
  width: number;
  depth: number;
  height: number;
  x: number;
  z: number;
  color: string;
}

export interface SceneSummary {
  objects: SceneObjectSummary[];
  /** Shape the next hand-built primitive will use. */
  activeShape: ShapeId;
  cameraRunning: boolean;
}

/** One-line plain-English scene description, shared so the model and the
 *  spoken reply describe the scene the same way. */
export function describeScene(scene: SceneSummary): string {
  if (scene.objects.length === 0) return 'The scene is empty.';
  const parts = scene.objects.map((o) => {
    const size = o.shape === 'sphere' || o.shape === 'cylinder'
      ? `${round(o.width)} wide`
      : `${round(o.width)} by ${round(o.depth)}`;
    const where = o.x === 0 && o.z === 0 ? 'at the center' : `at ${round(o.x)}, ${round(o.z)}`;
    return `a ${o.shape} ${size} and ${round(o.height)} tall ${where}`;
  });
  return `The scene has ${parts.length} object${parts.length === 1 ? '' : 's'}: ${parts.join('; ')}.`;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

/* -------------------------------------------------------------------------- */
/* Commands (model -> browser)                                                 */
/* -------------------------------------------------------------------------- */

/** A validated, ready-to-execute command. The browser only ever sees these. */
export type CadCommand =
  | { name: 'set_shape'; shape: ShapeId }
  | {
      name: 'add_shape';
      shape: ShapeId;
      width: number;
      depth: number;
      height: number;
      x: number;
      z: number;
      color: string | null;
    }
  | { name: 'remove_last' }
  | { name: 'clear_scene' }
  | { name: 'set_color'; color: string; target: 'last' | 'all' }
  | { name: 'describe_scene' }
  | { name: 'export_for_printing' }
  | { name: 'start_camera' }
  | { name: 'stop_camera' };

export type CadCommandName = CadCommand['name'];

/** A raw call as it arrives from the model, before validation. */
export interface RawToolCall {
  id?: string;
  name: string;
  args: Record<string, unknown>;
}

/** What the browser reports back so the model can narrate the outcome. */
export interface ToolOutcome {
  id?: string;
  name: string;
  ok: boolean;
  detail: string;
}

export type ValidationResult =
  | { ok: true; command: CadCommand }
  | { ok: false; error: string };

/**
 * Anything that can carry out a validated command. `src/main.ts` implements
 * this over `CadBuilder` / `CadScene`; keeping it an interface is what lets
 * `src/voice` stay free of CAD imports.
 */
export interface ToolExecutor {
  execute(command: CadCommand): string | Promise<string>;
}

/* -------------------------------------------------------------------------- */
/* Validation                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Coerce and range-check a model tool call. Out-of-range numbers are clamped
 * rather than rejected (this is a playground — "make it huge" should still
 * build something), but unknown names, shapes and colors are refused so the
 * model gets told instead of silently doing the wrong thing.
 */
export function validateToolCall(call: RawToolCall): ValidationResult {
  const args = call.args ?? {};
  switch (call.name) {
    case 'set_shape': {
      const shape = readShape(args.shape);
      if (!shape) return { ok: false, error: `Unknown shape. Use one of: ${SHAPE_IDS.join(', ')}.` };
      return { ok: true, command: { name: 'set_shape', shape } };
    }
    case 'add_shape': {
      const shape = readShape(args.shape);
      if (!shape) return { ok: false, error: `Unknown shape. Use one of: ${SHAPE_IDS.join(', ')}.` };
      const width = clamp(readNumber(args.width, 1), SIZE_MIN, SIZE_MAX);
      // Round footprints follow the width so a cylinder never comes out oval.
      const depth =
        shape === 'cylinder' || shape === 'sphere'
          ? width
          : clamp(readNumber(args.depth, width), SIZE_MIN, SIZE_MAX);
      const height = clamp(readNumber(args.height, 1), SIZE_MIN, SIZE_MAX);
      const color = args.color === undefined || args.color === null ? null : readColor(args.color);
      if (args.color !== undefined && args.color !== null && color === null) {
        return { ok: false, error: 'Unknown color. Use a name like "red" or a #rrggbb hex value.' };
      }
      return {
        ok: true,
        command: {
          name: 'add_shape',
          shape,
          width,
          depth,
          height,
          x: clamp(readNumber(args.x, 0), -POSITION_LIMIT, POSITION_LIMIT),
          z: clamp(readNumber(args.z, 0), -POSITION_LIMIT, POSITION_LIMIT),
          color,
        },
      };
    }
    case 'set_color': {
      const color = readColor(args.color);
      if (!color) {
        return { ok: false, error: 'Unknown color. Use a name like "red" or a #rrggbb hex value.' };
      }
      const target = args.target === 'all' ? 'all' : 'last';
      return { ok: true, command: { name: 'set_color', color, target } };
    }
    case 'remove_last':
    case 'clear_scene':
    case 'describe_scene':
    case 'export_for_printing':
    case 'start_camera':
    case 'stop_camera':
      return { ok: true, command: { name: call.name } };
    default:
      return { ok: false, error: `Unknown tool "${call.name}".` };
  }
}

function readShape(value: unknown): ShapeId | null {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  // Common spoken synonyms for the four primitives.
  const alias: Record<string, ShapeId> = {
    cube: 'box',
    square: 'box',
    block: 'box',
    rectangle: 'cuboid',
    slab: 'cuboid',
    tube: 'cylinder',
    circle: 'cylinder',
    ball: 'sphere',
    orb: 'sphere',
  };
  if (alias[normalized]) return alias[normalized];
  return (SHAPE_IDS as readonly string[]).includes(normalized) ? (normalized as ShapeId) : null;
}

function readColor(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (NAMED_COLORS[normalized]) return NAMED_COLORS[normalized];
  return /^#[0-9a-f]{6}$/.test(normalized) ? normalized : null;
}

function readNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/* -------------------------------------------------------------------------- */
/* Gemini function declarations                                                */
/* -------------------------------------------------------------------------- */

/** Minimal OpenAPI-ish schema node — structurally compatible with the Gemini
 *  SDK's `FunctionDeclaration` without importing it into the browser bundle. */
interface SchemaNode {
  type: 'OBJECT' | 'STRING' | 'NUMBER';
  description?: string;
  properties?: Record<string, SchemaNode>;
  required?: string[];
  enum?: string[];
}

export interface AgentFunctionDeclaration {
  name: CadCommandName;
  description: string;
  parameters?: SchemaNode;
}

const shapeParam: SchemaNode = {
  type: 'STRING',
  description: 'Which primitive: box, cuboid, cylinder or sphere.',
  enum: [...SHAPE_IDS],
};

export const AGENT_TOOLS: readonly AgentFunctionDeclaration[] = [
  {
    name: 'add_shape',
    description:
      'Build a new solid in the scene. Use this whenever the user asks for something to appear.',
    parameters: {
      type: 'OBJECT',
      properties: {
        shape: shapeParam,
        width: { type: 'NUMBER', description: `Width, or diameter for round shapes (${SIZE_MIN}-${SIZE_MAX}).` },
        depth: { type: 'NUMBER', description: 'Depth. Ignored for cylinders and spheres.' },
        height: { type: 'NUMBER', description: `Height (${SIZE_MIN}-${SIZE_MAX}).` },
        x: { type: 'NUMBER', description: `Left/right position, 0 is the center (-${POSITION_LIMIT} to ${POSITION_LIMIT}).` },
        z: { type: 'NUMBER', description: `Forward/back position, 0 is the center (-${POSITION_LIMIT} to ${POSITION_LIMIT}).` },
        color: { type: 'STRING', description: 'Optional color name or #rrggbb hex.' },
      },
      required: ['shape'],
    },
  },
  {
    name: 'set_shape',
    description:
      'Choose the primitive the user will build next with their hands. Use when they say "switch to cylinder" without asking you to build it.',
    parameters: {
      type: 'OBJECT',
      properties: { shape: shapeParam },
      required: ['shape'],
    },
  },
  {
    name: 'set_color',
    description: 'Recolor the most recent object, or every object when target is "all".',
    parameters: {
      type: 'OBJECT',
      properties: {
        color: { type: 'STRING', description: 'Color name or #rrggbb hex.' },
        target: { type: 'STRING', description: 'Either "last" or "all".', enum: ['last', 'all'] },
      },
      required: ['color'],
    },
  },
  { name: 'remove_last', description: 'Delete the most recently created object (undo).' },
  { name: 'clear_scene', description: 'Delete every object. Only when the user clearly asks to start over.' },
  { name: 'describe_scene', description: 'Read back what is currently in the scene.' },
  { name: 'export_for_printing', description: 'Download the scene as an STL file for 3D printing.' },
  { name: 'start_camera', description: 'Turn on the webcam and hand tracking.' },
  { name: 'stop_camera', description: 'Turn off the webcam and hand tracking.' },
];

/** System prompt: playful build-buddy, not a CAD manual. */
export const AGENT_SYSTEM_PROMPT = [
  'You are the voice of a playful 3D building playground. The user is standing in front of a webcam,',
  'building shapes with their hands, and talking to you at the same time.',
  '',
  'How to behave:',
  '- Keep spoken replies to one short sentence. You are heard, not read.',
  '- Prefer doing over asking. If a request is vague, pick something reasonable and build it,',
  '  then say what you made. Only ask a question if you truly cannot guess.',
  '- Be warm and a little bit fun. Never use CAD jargon like "extrude", "primitive" or "mesh".',
  '  Say "box", "tube", "ball", "taller", "wider".',
  '- The user can already build with their hands, so do not explain the gestures unless asked.',
  '- You get a picture of their screen each turn. Use it to judge sizes and free space.',
  '- Objects sit on a floor grid with the center at 0,0. Spread things out so they do not overlap.',
  '- Never call clear_scene unless the user clearly wants to start over.',
].join('\n');
