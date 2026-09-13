import { describe, expect, it } from "vitest";
import { stemmer } from "stemmer";

import { TextSummarizer } from "../src/lib/text_summarizer/text_summarizer.js";
import type { EmbeddingViewProps } from "../src/lib/embedding_view/embedding_view_api.js";

const rectangle = (x: number) => ({ xMin: x, xMax: x + 1, yMin: 0, yMax: 1 });
const regions = [[rectangle(0)], [rectangle(2)]];
const rows = (texts: (string | null | undefined)[], xs = texts.map((_, i) => i * 2 + 0.5)) => ({
  text: texts,
  x: xs,
  y: texts.map(() => 0.5),
});
const stems = (words: string[]) => words.map((word) => stemmer(word.toLowerCase()));

describe("BM25 region labels", () => {
  it("prefers multi-document support to an isolated word repeated many times", () => {
    for (const repeats of [1, 15, 10000]) {
      const summarizer = new TextSummarizer({ regions: [regions[0]] });
      summarizer.add(rows(["outlier ".repeat(repeats), "orchard", "orchard"], [0.5, 0.5, 0.5]));
      expect(summarizer.summarize()).toEqual([["orchard", "outlier"]]);
    }
  });

  it("keeps a shared theme ahead of a rare term found in only one of three documents", () => {
    const summarizer = new TextSummarizer({ regions: [regions[0]] });
    summarizer.add(rows(["orchard " + "outlier ".repeat(15), "orchard", "orchard"], [0.5, 0.5, 0.5]));
    expect(summarizer.summarize()).toEqual([["orchard", "outlier"]]);
  });

  it("does not let within-document repetition alter regional ranking or normalization", () => {
    const baseline = new TextSummarizer({ regions });
    const repeated = new TextSummarizer({ regions });
    const xs = [0.5, 0.5, 0.5, 2.5, 2.5, 2.5];
    const documents = [
      "orchard outlier",
      "orchard apple",
      "orchard apple",
      "marine coral",
      "marine coral",
      "marine tide",
    ];
    baseline.add(rows(documents, xs));
    repeated.add(
      rows(
        documents.map((text, i) => (i === 0 ? text + " outlier".repeat(10000) : text)),
        xs,
      ),
    );
    expect(repeated.summarize()).toEqual(baseline.summarize());
  });

  it("counts case and stem variants as a single regional document vote", () => {
    const summarizer = new TextSummarizer({ regions: [regions[0]] });
    summarizer.add(rows(["running runs RUNNING ".repeat(15), "orchard", "orchard"], [0.5, 0.5, 0.5]));
    expect(summarizer.summarize(1)).toEqual([["orchard"]]);
  });

  it("puts regional differences before a shared template without repeating same-level words", () => {
    const summarizer = new TextSummarizer({ regions });
    summarizer.add(
      rows([
        "common template document apple orchard harvest cider",
        "common template document ocean coral marine tide",
      ]),
    );
    const result = summarizer.summarize();
    expect(result[0]).toHaveLength(4);
    expect(result[1]).toHaveLength(4);
    expect(new Set(result.flat()).size).toBe(8);
    expect(result[0]).toEqual(expect.arrayContaining(["apple", "orchard", "harvest", "cider"]));
    expect(result[1]).toEqual(expect.arrayContaining(["ocean", "coral", "marine", "tide"]));
  });

  it("fills scarce candidates from local words and never duplicates a stem inside a label", () => {
    const summarizer = new TextSummarizer({ regions });
    summarizer.add(rows(["shared running runs", "shared running runs"]));
    const result = summarizer.summarize();
    for (const label of result) {
      expect(new Set(stems(label))).toEqual(new Set(["share", "run"]));
      expect(label).toHaveLength(2);
    }
  });

  it("allows the best word at both zoom levels, but diversifies within a level", () => {
    const texts = rows(["apple apple banana", "apple apple banana"]);
    const sameLevel = new TextSummarizer({ regions });
    const crossLevel = new TextSummarizer({ regions, levels: [0, 1] });
    sameLevel.add(texts);
    crossLevel.add(texts);
    expect(sameLevel.summarize(1)).toEqual([["apple"], ["banana"]]);
    expect(crossLevel.summarize(1)).toEqual([["apple"], ["apple"]]);
  });

  it("uses documents outside all regions for global document frequency", () => {
    const summarizer = new TextSummarizer({ regions: [regions[0]] });
    summarizer.add(rows(["apple zebra", "apple", "apple", "apple"], [0.5, 10, 11, 12]));
    expect(summarizer.summarize(1)).toEqual([["zebra"]]);
  });

  it("counts a repeated term once per document for IDF", () => {
    const short = new TextSummarizer({ regions: [regions[0]] });
    const long = new TextSummarizer({ regions: [regions[0]] });
    short.add(rows(["apple zebra", "apple"], [0.5, 10]));
    long.add(rows(["apple zebra", "apple ".repeat(10000)], [0.5, 10]));
    expect(long.summarize()).toEqual(short.summarize());
    expect(long.summarize()[0][0]).toBe("zebra");
  });

  it("saturates repeated template words instead of allowing raw frequency to dominate", () => {
    const summarizer = new TextSummarizer({ regions });
    summarizer.add(rows(["common ".repeat(10000) + "orchard", "common ".repeat(10000) + "coral"]));
    expect(summarizer.summarize(1)).toEqual([["orchard"], ["coral"]]);
  });

  it("handles unequal region sizes and long documents with finite, usable rankings", () => {
    const summarizer = new TextSummarizer({ regions });
    summarizer.add(
      rows(
        [...Array<string>(100).fill("common orchard harvest"), "common ".repeat(1000) + "coral marine"],
        [...Array<number>(100).fill(0.5), 2.5],
      ),
    );
    const result = summarizer.summarize();
    expect(result[0][0]).not.toBe("common");
    expect(result[1][0]).not.toBe("common");
    expect(result[0]).toHaveLength(3);
    expect(result[1]).toHaveLength(3);
  });

  it("ignores empty text, nonwords, numbers, and case-insensitive custom stopwords", () => {
    const summarizer = new TextSummarizer({ regions: [regions[0]], stopWords: ["IGNORE"] });
    summarizer.add(rows([null, undefined, "", "  ", "!!! 123 １２３ Ignore ignore apple"], Array(5).fill(0.5)));
    expect(summarizer.summarize()).toEqual([["apple"]]);
  });

  it("supports Chinese navigation words", () => {
    const summarizer = new TextSummarizer({ regions, stopWords: ["公共", "模板"] });
    summarizer.add(rows(["公共 模板 苹果 果园", "公共 模板 海洋 珊瑚"]));
    const result = summarizer.summarize();
    expect(result[0]).toEqual(expect.arrayContaining(["苹果", "果园"]));
    expect(result[1]).toEqual(expect.arrayContaining(["海洋", "珊瑚"]));
    expect(result.flat()).not.toContain("模板");
  });

  it("uses the most frequent surface form with deterministic lexical ties", () => {
    const summarizer = new TextSummarizer({ regions: [regions[0]] });
    summarizer.add(rows(["runs running running RUNNING RUNNING"], [0.5]));
    expect(summarizer.summarize()).toEqual([["RUNNING"]]);
  });

  it("returns the same result for streamed chunks and reordered documents", () => {
    const data = ["common apple orchard", "common ocean coral", "common apple harvest"];
    const all = new TextSummarizer({ regions });
    const streamed = new TextSummarizer({ regions });
    const reversed = new TextSummarizer({ regions });
    all.add(rows(data, [0.5, 2.5, 0.5]));
    data.forEach((text, i) => streamed.add(rows([text], [i === 1 ? 2.5 : 0.5])));
    reversed.add(rows([...data].reverse(), [0.5, 2.5, 0.5]));
    expect(streamed.summarize()).toEqual(all.summarize());
    expect(reversed.summarize()).toEqual(all.summarize());
    expect(all.summarize()).toEqual(all.summarize());
  });

  it("does not multiply document frequency when a point belongs to overlapping regions", () => {
    const single = new TextSummarizer({ regions: [regions[0]] });
    const overlapping = new TextSummarizer({ regions: [regions[0], regions[0]], levels: [0, 1] });
    const data = rows(["apple zebra", "apple"], [0.5, 10]);
    single.add(data);
    overlapping.add(data);
    expect(overlapping.summarize()).toEqual([single.summarize()[0], single.summarize()[0]]);
    expect(overlapping.summarize()[0][0]).toBe("zebra");
  });

  it("excludes points outside rectangles even when they share a spatial bin", () => {
    const summarizer = new TextSummarizer({ regions: [[rectangle(0)], [rectangle(1000)]] });
    summarizer.add(rows(["inside", "outside"], [0.5, 1.01]));
    expect(summarizer.summarize()).toEqual([["inside"], []]);
  });

  it("handles no regions, empty regions, and nonpositive limits", () => {
    expect(new TextSummarizer({ regions: [] }).summarize()).toEqual([]);
    const summarizer = new TextSummarizer({ regions: [[], regions[0]] });
    expect(summarizer.summarize()).toEqual([[], []]);
    summarizer.add(rows(["apple"], [0.5]));
    expect(summarizer.summarize()).toEqual([[], ["apple"]]);
    expect(summarizer.summarize(0)).toEqual([[], []]);
  });

  it("keeps legacy one-argument label callbacks compatible with the optional level context", async () => {
    const callback: NonNullable<EmbeddingViewProps["queryClusterLabels"]> = async (clusters) =>
      clusters.map(() => "legacy");
    expect(await callback(regions, { levels: [0, 1] })).toEqual(["legacy", "legacy"]);
  });
});
