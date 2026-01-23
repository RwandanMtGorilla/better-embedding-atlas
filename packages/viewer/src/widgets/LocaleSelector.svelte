<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { locale, setLocale, supportedLocales, type LocaleCode } from "../i18n";
  import Select from "./Select.svelte";
</script>

<Select
  value={$locale as LocaleCode}
  onChange={(v) => setLocale(v as LocaleCode)}
  options={supportedLocales.map((l) => ({ value: l.code, label: l.name }))}
/>
