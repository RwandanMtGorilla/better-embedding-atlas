# Automatic navigation labels

Automatic labels rank words from each density region using a BM25-style score:

`documentIDF * regionIDF * coverage^2 * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / averageLength))`

Here `k1 = 1.2`, `b = 0.75`, and both IDFs use
`log(1 + (N - df + 0.5) / (df + 0.5))`. Document IDF counts each normalized
word stem once per valid document across the full input table, including documents
outside the label regions. Region IDF and average length use nonempty regions at
the same zoom level. `tf` counts regional documents containing the stem, and
`coverage = tf / valid regional document count`. Region length is the sum of
per-document unique stem counts. Repeating a word within one document changes
neither its score nor the length normalization. Squared coverage downweights
isolated-document words, balancing their global rarity against regional support.
For three documents, a word supported by only one gets a coverage weight of 1/9,
versus 4/9 for two documents and 1 for all three. Rare words remain available as
supplementary candidates; coverage is a soft weight, not a majority cutoff.
This is an adaptation for navigation labels, not a search-query BM25 score.

Words are segmented with `Intl.Segmenter`, filtered for stopwords, numbers, and
nonwords, then grouped by lowercase stems. Each region displays its most frequent
original spelling, with lexical tie-breaking. Empty or missing text contributes
no statistics. Streaming chunks does not change the result.

Each zoom level allocates up to four words per region in rounds. Candidates with
higher scores win conflicts; ties use stem order followed by region index. A stem
is used only once per level during this pass. Afterwards, regions with fewer than
four words may reuse topic words from their own text. Thus completely identical
documents may still have identical labels. Empty regions have no label.

`queryClusterLabels(clusters, context?)` receives optional `{ levels: number[] }`
in cluster order. Existing callbacks accepting only `clusters` remain compatible.
The built-in worker accepts the same optional `levels` array in its creation
options; omitted levels default to zero. Coarse and fine labels can reuse words.
Custom labels and UMAP coordinates are unchanged. Automatic-label cache version 3
invalidates labels generated with the previous ranking algorithm.

Run `npm run test -w @embedding-atlas/component` for regression scenarios.

For visual verification, run `npm run dev -w @embedding-atlas/component` and open
`/test/fixtures/navigation-labels.html`. The deterministic 2,400-point sample uses
three topics with shared template words, the real clustering/keyword worker, and
a seeded version-1 cache entry. Confirm distinct topic labels before and after
zooming, both levels in the diagnostic output, and `oldCacheHits: 0` with
`callbackCalls: 1`.
