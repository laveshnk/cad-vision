/** Public API of the voice agent (hands-free speech -> CAD commands). */
export { VoiceAgent } from './VoiceAgent';
export type { VoiceAgentOptions, VoiceAgentCallbacks, VoiceState } from './VoiceAgent';
export { TalkGate } from './TalkGate';
export type { TalkGateOptions, TalkGateEvent } from './TalkGate';
export { PresageFrames } from './PresageFrames';
export type { PresageFramesOptions, PresageFramesCallbacks, PresageStatus } from './PresageFrames';
export { Transcriber } from './Transcriber';
export type { TranscriberCallbacks } from './Transcriber';
export { Speaker } from './Speaker';
export type { SpeakerCallbacks } from './Speaker';
export {
  AGENT_TOOLS,
  AGENT_SYSTEM_PROMPT,
  NAMED_COLORS,
  SHAPE_IDS,
  describeScene,
  validateToolCall,
} from '../../shared/agentTools';
export type {
  CadCommand,
  CadCommandName,
  RawToolCall,
  SceneObjectSummary,
  SceneSummary,
  ShapeId,
  ToolExecutor,
  ToolOutcome,
} from '../../shared/agentTools';
