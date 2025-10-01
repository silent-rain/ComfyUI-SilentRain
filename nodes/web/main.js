import init, { run } from './pkg/web.js';


async function main() {
	await init();
	// await run();
	console.log("Hello from Rust!");
}
main();
