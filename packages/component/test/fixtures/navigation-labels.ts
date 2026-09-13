import { cacheKeyForObject } from "../../src/lib/utils.js";
import { EmbeddingView } from "../../src/lib/embedding_view/embedding_view_api.js";
import { median, stdev } from "../../src/lib/embedding_view/statistics.js";
import {
  textSummarizerAdd,
  textSummarizerCreate,
  textSummarizerDestroy,
  textSummarizerSummarize,
} from "../../src/lib/embedding_view/worker/index.js";

const topics = ["apple orchard harvest cider", "ocean coral marine tide", "software compiler debug runtime"];
const x: number[] = [];
const y: number[] = [];
const text: string[] = [];
for (let topic = 0; topic < topics.length; topic++) {
  for (let i = 0; i < 800; i++) {
    const angle = i * 2.399963229728653;
    const radius = 0.8 * Math.sqrt((i + 0.5) / 800);
    x.push((topic - 1) * 3 + radius * Math.cos(angle));
    y.push((topic === 1 ? 1 : -1) + radius * Math.sin(angle));
    text.push(`common template document ${topics[topic]}`);
  }
}
const data = { x: new Float32Array(x), y: new Float32Array(y) };
const viewport = {
  x: median(data.x),
  y: median(data.y),
  scale: (1 / (Math.max(stdev(data.x), stdev(data.y), 1e-3) * 3)) * 0.95,
};
const oldKey = await cacheKeyForObject({
  autoLabel: { version: 1, viewport, stopWords: undefined, densityThreshold: undefined },
});
const cache = new Map<string, unknown>([[oldKey, [{ x: 0, y: 0, text: "OLD CACHE LABEL" }]]]);
let oldCacheHits = 0;
let callbackCalls = 0;
const status = document.querySelector<HTMLPreElement>("#status")!;
const view = new EmbeddingView(document.querySelector<HTMLDivElement>("#view")!, {
  data,
  width: 1000,
  height: 550,
  cache: {
    get: async (key) => {
      if (key === oldKey) oldCacheHits++;
      return cache.get(key);
    },
    set: async (key, value) => {
      cache.set(key, value);
    },
  },
  queryClusterLabels: async (regions, context) => {
    callbackCalls++;
    const summarizer = await textSummarizerCreate({ regions, levels: context?.levels });
    try {
      await textSummarizerAdd(summarizer, { ...data, text });
      const words = await textSummarizerSummarize(summarizer);
      status.textContent = JSON.stringify(
        {
          oldCacheHits,
          callbackCalls,
          labels: words.map((words, i) => ({ level: context?.levels[i], words })),
        },
        null,
        2,
      );
      return words.map((words) =>
        words.length ? words.slice(0, 2).join("-") + "\n" + words.slice(2).join("-") : null,
      );
    } finally {
      await textSummarizerDestroy(summarizer);
    }
  },
});
document.querySelector<HTMLButtonElement>("#zoom-in")!.onclick = () => {
  view.update({ viewportState: { ...viewport, scale: viewport.scale * 2 } });
};
document.querySelector<HTMLButtonElement>("#reset")!.onclick = () => {
  view.update({ viewportState: viewport });
};
