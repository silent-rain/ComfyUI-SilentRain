// ComfyUI-SilentRain Web 入口

function loadReactBundle() {
	try {
		// 通过 ComfyUI 静态资源路径加载（extensions/<package>/...）
		// WEB_DIRECTORY 指向本目录，因此 dist/silentrain.bundle.js 是相对路径
		import('./dist/silentrain.bundle.js');
	} catch (e) {
		console.warn(
			'[SilentRain] React bundle not found, skip. Run scripts/build_web_react.sh to build it.',
			e,
		);
	}
}

function main() {
	loadReactBundle();
}

main();
