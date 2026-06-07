import React from "react";
import { NODE_EXTENSIONS } from "./nodes";


export default function App() {
    const app = window.app!;

    const names: string[] = [];
    for (const ext of NODE_EXTENSIONS) {
        try {
            // 注册节点扩展
            app.registerExtension(ext);

            names.push(ext.name);
        } catch (e) {
            console.error(`[SilentRain] failed to register ${ext.name}`, e);
        }
    }

    console.log(`[SilentRain] React extensions registered: ${names.join(', ')}`);

    return (
        <>
        </>
    );
}
