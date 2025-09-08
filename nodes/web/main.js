import init, { register_extension } from './pkg/web.js';


async function run() {
	await init();
	await register_extension();
	console.log("Hello from Rust!");
}
run();
