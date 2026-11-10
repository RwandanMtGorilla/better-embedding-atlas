import { AsyncDuckDB, VoidLogger } from "@duckdb/duckdb-wasm";
import duckdbWorker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdbWasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import { clausePoint, Coordinator, Priority, Selection, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingViewMosaic } from "../../src/lib/embedding_view/embedding_view_mosaic_api.js";
import {
  LABEL_TILE_MAP_SIZE,
  labelTileAnalysis,
  labelTilesWithNeighbors,
  labelViewportBounds,
  localLabelGrid,
  localLabelLevel,
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
const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
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

function queriesTile(sql: string, tile: LabelTile) {
  const { bounds } = labelTileAnalysis(tile);
  return (
    sql.includes(`"x" >= ${bounds.xMin}`) &&
    sql.includes(`"x" < ${bounds.xMax}`) &&
    sql.includes(`"y" >= ${bounds.yMin}`) &&
    sql.includes(`"y" < ${bounds.yMax}`)
  );
}

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
  const histogramQueries: { sql: string; priority: number; empty: boolean }[] = [];
  const textQueries: { sql: string; priority: number }[] = [];
  const coordinator = new Coordinator(connector);
  const request = coordinator.manager.request.bind(coordinator.manager);
  coordinator.manager.request = (query, priority = Priority.Normal) => {
    const result = request(query, priority);
    const sql = String(query.query);
    if (sql.includes('AS "bin_count"')) {
      const entry = { sql, priority, empty: false };
      histogramQueries.push(entry);
      void result.then(
        (data) => {
          entry.empty = (data as { numRows: number }).numRows === 0;
        },
        () => {},
      );
    }
    if (sql.includes("ORDER BY") && sql.includes('"text"')) textQueries.push({ sql, priority });
    return result;
  };
  const foregroundCount = () => histogramQueries.filter(({ priority }) => priority === Priority.Normal).length;

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
  const middle = { ...deep, scale: initial.scale * 8 };
  const approach = { ...target, scale: initial.scale * 2 };
  const empty = { ...target, x: 500, y: 500 };
  const tiles = (state: ViewportState) => {
    const level = localLabelLevel(initial.scale, state.scale)!;
    return visibleLabelTiles(localLabelGrid(initial.scale, level), labelViewportBounds(state, width, height));
  };
  const required = (state: ViewportState) => labelTilesWithNeighbors(tiles(state));
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

  async function showWarmed(state: ViewportState, since: number, correctLabels: () => boolean, action: string) {
    const needed = [...required(state).values()];
    await until(() => {
      const queries = histogramQueries.slice(since);
      // The next histogram starts only after the previous tile finishes clustering and summarizing.
      const complete = queries.filter(({ empty }, i) => empty || i < queries.length - 1);
      return (
        needed.every((tile) => complete.some(({ sql }) => queriesTile(sql, tile))) &&
        complete.some(({ sql, priority }) => priority === Priority.Low && needed.some((tile) => queriesTile(sql, tile)))
      );
    }, `${action}所需的瓦片已预热`);
    const beforeSwitch = histogramQueries.length;
    view.update({ viewportState: state });
    await nextFrame();
    check(correctLabels(), `${action}首帧显示正确标签`);
    await pause(350);
    check(
      histogramQueries.slice(beforeSwitch).every(({ sql }) => needed.every((tile) => !queriesTile(sql, tile))),
      `${action}后目标瓦片的新增查询数为零`,
    );
  }

  await until(() => hasLabel("coral"), "概览显示海洋主题");
  check(histogramQueries.length === 0, "概览没有分块查询");
  for (let i = 0; i < 8; i++) {
    view.update({ viewportState: i % 2 === 0 ? deep : empty });
    await pause(85);
  }
  check(histogramQueries.length === 0, "连续移动期间暂停局部查询");
  view.update({ viewportState: west });
  await until(
    () => foregroundCount() === required(west).size && hasLabel("pine") && hasLabel("apple"),
    "快速缩放只生成最终西侧视野所需的块",
  );
  const westLabels = snapshot(west);
  const westOverlap = inside(westLabels, target);
  check(westOverlap.length > 0, "西侧和目标视野有共享标签");

  await showWarmed(
    target,
    0,
    () =>
      hasLabel("glacier") &&
      same(inside(snapshot(target), west), westOverlap) &&
      host.querySelectorAll("svg text:not([data-label-id] text)").length === 0,
    "跨块平移",
  );
  const targetLabels = snapshot(target);
  check(same(inside(targetLabels, west), westOverlap), "新块加载后，共享分区 ID、词序和坐标保持一致");
  check(targetLabels.filter(({ words }) => words.includes("glacier")).length === 1, "跨边界的冰川簇只显示一个标签");

  view.update({ viewportState: east });
  await until(() => hasLabel("coral") && !hasLabel("pine"), "向东平移时海洋主题进入、松林主题离开");
  check(same(inside(snapshot(east), target), inside(targetLabels, east)), "向东平移也保留重叠区域的标签");
  const eastCount = foregroundCount();
  view.update({ viewportState: target });
  await until(() => same(snapshot(target), targetLabels), "从东侧进入同一视野时复现原标签");
  check(foregroundCount() === eastCount, "相同地图块复用内存结果");
  view.update({ config: { pointSize: 4 } });
  await pause(350);
  check(same(snapshot(target), targetLabels) && foregroundCount() === eastCount, "调整散点大小不重新生成关键词");

  await showWarmed(
    middle,
    0,
    () => hasLabel("apple") && snapshot(middle).every(({ id }) => id.startsWith("3:")),
    "放大一档",
  );
  view.update({ viewportState: deep });
  await until(() => hasLabel("apple") && hasLabel("cider"), "16 倍视野区分苹果与苹果酒子主题");
  check(
    snapshot(deep).some(({ words }) => !targetLabels.some((label) => label.words === words)),
    "深入一档得到新词",
  );
  view.update({ viewportState: target });
  await until(() => same(snapshot(target), targetLabels), "缩回原档后复现原词和位置");

  const approachStart = histogramQueries.length;
  view.destroy();
  view = open(approach);
  await until(() => histogramQueries.length > approachStart, "概览在 2 倍开始预热");
  check(snapshot(approach).length === 0, "2 倍视图仍显示概览层");
  await showWarmed(target, approachStart, () => same(snapshot(target), targetLabels), "从概览进入局部");

  const middleStart = histogramQueries.length;
  view.destroy();
  view = open(middle);
  await showWarmed(target, middleStart, () => same(snapshot(target), targetLabels), "缩小一档");
  check(
    textQueries.some(({ priority }) => priority === Priority.Low),
    "预热文本查询使用低优先级",
  );

  const orchardTile = tiles(target).find((tile) => tile.column === -1 && tile.row === 0)!;
  const isOrchardQuery = ({ sql }: { sql: string }) => queriesTile(sql, orchardTile);
  const textQuery = textQueries.find(isOrchardQuery)?.sql;
  const histogramQuery = histogramQueries.find(isOrchardQuery)?.sql;
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

  const beforeEmpty = foregroundCount();
  view.update({ viewportState: empty });
  await until(
    () => foregroundCount() >= beforeEmpty + required(empty).size && labels().length === 0,
    "空视野完成查询后仍留空",
  );
  view.update({ viewportState: initial });
  await until(() => labels().length > 0 && snapshot(initial).length === 0, "缩回 4 倍以下恢复概览");
  const emptyCount = foregroundCount();
  view.update({ viewportState: empty });
  await pause(350);
  check(labels().length === 0 && foregroundCount() === emptyCount, "重复访问空块也命中缓存");

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
