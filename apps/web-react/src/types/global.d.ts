import type { ComfyApp } from '@comfyorg/comfyui-frontend-types';

declare global {
  interface Window {
    app?: ComfyApp;
  }
}
