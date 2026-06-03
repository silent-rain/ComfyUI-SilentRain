// ComfyUI-SilentRain Web 入口
// 1) 启动原有 wasm 模块
// 2) 加载 React bundle（包含所有现代化 UI 节点：ParamHub / ParamPort 等）
import init from './pkg/web.js';

async function loadReactBundle() {
	try {
		// 通过 ComfyUI 静态资源路径加载（extensions/<package>/...）
		// WEB_DIRECTORY 指向本目录，因此 dist/silentrain.bundle.js 是相对路径
		await import('./dist/silentrain.bundle.js');
	} catch (e) {
		console.warn(
			'[SilentRain] React bundle not found, skip. Run scripts/build_web_react.sh to build it.',
			e,
		);
	}
}

async function main() {
	await init();
	await loadReactBundle();
	console.log('Hello from Rust!');
}

main();
