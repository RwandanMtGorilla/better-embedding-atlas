import { AsyncDuckDB, VoidLogger } from "@duckdb/duckdb-wasm";
import duckdbWorker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdbWasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import { clausePoint, Coordinator, Selection, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingViewMosaic } from "../../src/lib/embedding_view/embedding_view_mosaic_api.js";
import {
  LABEL_TILE_MAP_SIZE,
  labelTileAnalysis,
  labelTileNeighbors,
  labelViewportBounds,
  localLabelGrid,
  visibleLabelTiles,
  type LabelTile,
} from "../../src/lib/embedding_view/local_label_tiles.js";
import { queryApproximateDensity } from "../../src/lib/embedding_view/mosaic_client.js";
import { findLabelTileClusters } from "../../src/lib/embedding_view/worker/index.js";
import type { ViewportState } from "../../src/lib/utils.js";
import { Viewport } from "../../src/lib/viewport_utils.js";

const host = document.querySelector<HTMLDivElement>("#view")!;
const status = document.querySelector<HTMLPreElement>("#status")!;
const report: string[] = [];
const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const width = 1000;
const height = 350;

function check(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
  report.push(message);
  status.textContent = report.join("\n");
}

async function until(condition: () => boolean, message: string) {
  const deadline = performance.now() + 15000;
  while (!condition() && performance.now() < deadline) {
    await pause(25);
  }
  check(condition(), message);
}

function snapshot(state: ViewportState) {
  const viewport = new Viewport(state, width, height);
  return Array.from(host.querySelectorAll<SVGGElement>("svg [data-label-id]"), (element) => {
    const [px, py] = element.getAttribute("transform")!.slice("translate(".length, -1).split(",").map(Number);
    const { x, y } = viewport.coordinateAtPixel(px, py);
    return {
      id: element.dataset.labelId!,
      x: Number(x.toFixed(9)),
      y: Number(y.toFixed(9)),
      words: Array.from(element.querySelectorAll("text"), (line) => line.textContent).join("\n"),
    };
  }).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

type Snapshot = ReturnType<typeof snapshot>;
const same = (a: Snapshot, b: Snapshot) => JSON.stringify(a) === JSON.stringify(b);
const inside = (labels: Snapshot, state: ViewportState) => {
  const { xMin, xMax, yMin, yMax } = labelViewportBounds(state, width, height);
  return labels.filter(({ x, y }) => x >= xMin && x < xMax && y >= yMin && y < yMax);
};

async function checkDensityTies(tile: LabelTile) {
  const signal = new AbortController().signal;
  async function cluster(spacing: number, offset: number) {
    const size = LABEL_TILE_MAP_SIZE;
    const grid = new Float64Array(size * size);
    for (const [x, y] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 0],
    ]) {
      grid[(220 + y * spacing) * size + 220 + offset + x * spacing] = 120;
    }
    return JSON.stringify(await findLabelTileClusters(grid, tile, undefined, signal));
  }
  const first = await cluster(9, 0);
  let matches = true;
  for (let i = 1; i <= 4; i++) {
    await cluster(7, i);
    matches = (await cluster(9, 0)) === first && matches;
  }
  check(matches && first !== "[]", "并列密度峰在插入其他聚类任务后仍得到相同分区");
}

async function run() {
  const duckdb = new AsyncDuckDB(new VoidLogger(), new Worker(duckdbWorker));
  await duckdb.instantiate(duckdbWasm);
  const connector = wasmConnector({ duckdb });
  const histogramQueries: string[] = [];
  const textQueries: string[] = [];
  const coordinator = new Coordinator(connector, {
    logger: {
      ...console,
      debug(message: string, query?: { sql?: string }) {
        if (message !== "Query" || query?.sql == null) return;
        if (query.sql.includes('AS "bin_count"')) histogramQueries.push(query.sql);
        if (query.sql.includes("ORDER BY") && query.sql.includes('"text"')) textQueries.push(query.sql);
      },
    },
  });
  coordinator.manager.logQueries(true);

  const values: string[] = [];
  function patch(x: number, y: number, words: string, count: number, radius = 0.024, keep = true) {
    for (let i = 0; i < count; i++) {
      const angle = i * 2.399963229728653;
      const distance = radius * Math.sqrt((i + 0.5) / count);
      values.push(`(${x + distance * Math.cos(angle)}, ${y + distance * Math.sin(angle)}, '${words}', ${keep})`);
    }
  }
  patch(-2.45, 0.72, "orchard apple harvest gala", 120);
  patch(-2.3, 0.84, "orchard cider press barrel", 85);
  patch(-6.5, 0.7, "forest pine needle evergreen", 125);
  patch(-0.012, 1.15, "seam glacier frost snow", 120, 0.06);
  patch(5.5, 0.68, "ocean coral marine tide", 260);
  patch(11, 0.68, "distant quartz stone mineral", 90);
  patch(-3, 4.5, "offscreen saffron spice paprika", 32);
  patch(-2.45, 0.8, "excluded zircon foil bronze", 65, 0.024, false);
  await coordinator.exec(`CREATE TABLE sample AS SELECT * FROM (VALUES ${values.join(",")}) t(x, y, text, keep)`);

  const source = { table: "sample", x: "x", y: "y" };
  const { centerX, centerY, scaler } = await queryApproximateDensity(coordinator, source);
  const initial = { x: centerX, y: centerY, scale: scaler * 0.95 };
  const scale = initial.scale * 4;
  const half = 1 / scale;
  const target = { x: -0.25 * half, y: 0.8, scale };
  const west = { ...target, x: -1.18 * half };
  const east = { ...target, x: 0.35 * half };
  const deep = { x: -2.37, y: 0.78, scale: initial.scale * 16 };
  const empty = { ...target, x: 500, y: 500 };
  const grid = localLabelGrid(initial.scale, 2);
  const tiles = (state: ViewportState) => visibleLabelTiles(grid, labelViewportBounds(state, width, height));
  const required = (state: ViewportState) =>
    new Set(
      tiles(state)
        .flatMap(labelTileNeighbors)
        .map(({ key }) => key),
    );
  await checkDensityTies(tiles(target)[0]);
  const filter = Selection.crossfilter();
  const filterSource = {};
  const keep = (value: boolean) => filter.update(clausePoint("keep", value, { source: filterSource }));
  keep(true);
  const open = (viewportState?: ViewportState) =>
    new EmbeddingViewMosaic(host, { ...source, coordinator, filter, text: "text", width, height, viewportState });
  let view = open();
  const labels = () => Array.from(host.querySelectorAll("svg text"), (element) => element.textContent ?? "");
  const hasLabel = (word: string) => labels().some((line) => line.includes(word));

  await until(() => hasLabel("coral"), "概览显示海洋主题");
  check(histogramQueries.length === 0, "概览没有分块查询");
  view.update({ viewportState: deep });
  await pause(70);
  view.update({ viewportState: empty });
  await pause(70);
  view.update({ viewportState: west });
  await until(
    () => histogramQueries.length === required(west).size && hasLabel("pine") && hasLabel("apple"),
    "快速缩放只生成最终西侧视野所需的块",
  );
  const westLabels = snapshot(west);
  const westOverlap = inside(westLabels, target);
  check(westOverlap.length > 0, "西侧和目标视野有共享标签");

  view.update({ viewportState: target });
  await pause(45);
  check(
    same(inside(snapshot(target), west), westOverlap) &&
      host.querySelectorAll("svg text:not([data-label-id] text)").length === 0,
    "跨块移动时共享标签仍在原坐标，且没有闪回概览",
  );
  const neededAfterTarget = new Set([...required(west), ...required(target)]);
  await until(
    () => histogramQueries.length === neededAfterTarget.size && hasLabel("glacier"),
    "跨块后显示新进入的冰川主题",
  );
  const targetLabels = snapshot(target);
  check(same(inside(targetLabels, west), westOverlap), "新块加载后，共享分区 ID、词序和坐标保持一致");
  check(targetLabels.filter(({ words }) => words.includes("glacier")).length === 1, "跨边界的冰川簇只显示一个标签");

  view.update({ viewportState: east });
  await until(() => hasLabel("coral") && !hasLabel("pine"), "向东平移时海洋主题进入、松林主题离开");
  check(same(inside(snapshot(east), target), inside(targetLabels, east)), "向东平移也保留重叠区域的标签");
  const eastCount = histogramQueries.length;
  view.update({ viewportState: target });
  await until(() => same(snapshot(target), targetLabels), "从东侧进入同一视野时复现原标签");
  check(histogramQueries.length === eastCount, "相同地图块复用内存结果");
  view.update({ config: { pointSize: 4 } });
  await pause(350);
  check(same(snapshot(target), targetLabels) && histogramQueries.length === eastCount, "调整散点大小不重新生成关键词");

  view.update({ viewportState: deep });
  await until(() => hasLabel("apple") && hasLabel("cider"), "16 倍视野区分苹果与苹果酒子主题");
  check(
    snapshot(deep).some(({ words }) => !targetLabels.some((label) => label.words === words)),
    "深入一档得到新词",
  );
  view.update({ viewportState: target });
  await until(() => same(snapshot(target), targetLabels), "缩回原档后复现原词和位置");

  const coldCount = histogramQueries.length;
  view.destroy();
  view = open(target);
  await until(
    () => histogramQueries.length > coldCount && same(snapshot(target), targetLabels),
    "销毁重开后从空缓存重新生成相同结果",
  );

  const orchardTile = tiles(target).find((tile) => tile.column === -1 && tile.row === 0)!;
  const { bounds } = labelTileAnalysis(orchardTile);
  const isOrchardQuery = (query: string) =>
    query.includes(`"x" >= ${bounds.xMin}`) && query.includes(`"y" >= ${bounds.yMin}`);
  const textQuery = textQueries.find(isOrchardQuery);
  const histogramQuery = histogramQueries.find(isOrchardQuery);
  check(textQuery != null && histogramQuery != null, "同一块使用同一范围查询密度和文本");
  const documents = await connector.query({ type: "arrow", sql: textQuery! });
  const corpus = Array.from(documents.getChild("text")!.toArray(), String);
  const histogram = await connector.query({ type: "arrow", sql: histogramQuery! });
  const total = Array.from(histogram.getChild("bin_count")!.toArray(), Number).reduce((sum, value) => sum + value, 0);
  check(
    total === corpus.length &&
      corpus.some((text) => text.includes("offscreen")) &&
      corpus.every((text) => !/excluded|distant|ocean/.test(text)) &&
      !hasLabel("saffron"),
    "区内屏外文本进入语料，区外和被筛选文本不参与",
  );

  keep(false);
  await until(() => hasLabel("zircon") && !hasLabel("orchard") && !hasLabel("glacier"), "切换筛选清缓存并重算");
  view.update({ config: { autoLabelStopWords: ["zircon"] } });
  await until(() => hasLabel("bronze") && !hasLabel("zircon"), "更改停用词后重新生成");

  const beforeEmpty = histogramQueries.length;
  view.update({ viewportState: empty });
  await until(
    () => histogramQueries.length >= beforeEmpty + required(empty).size && labels().length === 0,
    "空视野完成查询后仍留空",
  );
  view.update({ viewportState: initial });
  await until(() => labels().length > 0 && snapshot(initial).length === 0, "缩回 4 倍以下恢复概览");
  const emptyCount = histogramQueries.length;
  view.update({ viewportState: empty });
  await pause(350);
  check(labels().length === 0 && histogramQueries.length === emptyCount, "重复访问空块也命中缓存");

  await coordinator.exec("CREATE TABLE replacement AS SELECT x, y, 'replacement sprout' AS text, keep FROM sample");
  view.update({ table: "replacement", viewportState: target });
  await until(() => hasLabel("replacement") && !hasLabel("bronze"), "切换数据源后按新文本生成标签");

  view.update({ labels: [{ x: target.x, y: target.y, text: "manual label" }] });
  await until(() => hasLabel("manual label") && labels().length === 1, "深层视野优先显示手动标签");
  view.update({ labels: null, config: { autoLabelEnabled: false } });
  await pause(350);
  check(labels().length === 0, "关闭自动标签后局部视野留空");
}

try {
  await run();
  status.dataset.status = "pass";
} catch (error) {
  status.dataset.status = "fail";
  status.textContent = [...report, String(error)].join("\n");
  throw error;
}
