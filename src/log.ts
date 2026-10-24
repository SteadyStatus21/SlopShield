import { DEBUG } from './config';

export const log = (...args: unknown[]) => {
  if (DEBUG) console.log('[slopshield]', ...args);
};
export const warn = (...args: unknown[]) => {
  if (DEBUG) console.warn('[slopshield]', ...args);
};
