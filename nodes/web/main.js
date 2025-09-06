import { app } from "../../scripts/app.js";
import init, { greet } from './pkg/web.js';


app.registerExtension({ 
	name: "a.unique.name.for.a.useless.extension",
	async setup() { 
		alert("Setup complete!")
	},
})


async function run() {
    await init();
    console.log(greet("World"));
}
run();
