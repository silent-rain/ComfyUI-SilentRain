// ComfyUI-SilentRain React Bundle 入口
//
// 通过 nodes/web/main.js 在 ComfyUI 环境中加载，
// 注册所有 React 节点扩展。
// import { app } from "../../../scripts/app.js";
import App from './App';

// 等待 ComfyUI 环境准备就绪
function ready(cb: () => void) {
  // window.app = app;
  if (typeof window === 'undefined') return;
  if (window.app) {
    cb();
    return;
  }
  let attempts = 0;
  const timer = setInterval(() => {
    attempts++;
    if (window.app) {
      clearInterval(timer);
      cb();
    } else if (attempts > 200) {
      clearInterval(timer);
      console.error('[SilentRain] window.app not ready after 20s, abort');
    }
  }, 100);
}

ready(() => {
  console.log('[SilentRain] App initialized');
  App()
});
