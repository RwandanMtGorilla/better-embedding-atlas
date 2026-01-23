// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { mount } from "svelte";

import "../app.css";
import { initI18n } from "../i18n";

import App from "./Entrypoint.svelte";

// Initialize i18n before mounting the app
initI18n();

const app = mount(App, { target: document.getElementById("app")! });

export default app;
