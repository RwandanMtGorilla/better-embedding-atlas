// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { init, register, getLocaleFromNavigator, locale, _ } from "svelte-i18n";

// Register locale files
register("en", () => import("./locales/en.json"));
register("zh-CN", () => import("./locales/zh-CN.json"));

// Supported locales
export const supportedLocales = [
  { code: "en", name: "English" },
  { code: "zh-CN", name: "简体中文" },
] as const;

export type LocaleCode = (typeof supportedLocales)[number]["code"];

// LocalStorage key for persisting user preference
const LOCALE_STORAGE_KEY = "embedding-atlas-locale";

/**
 * Get initial locale based on:
 * 1. User's saved preference in localStorage
 * 2. Browser language setting
 * 3. Fallback to English
 */
function getInitialLocale(): string {
  // Check localStorage first
  if (typeof localStorage !== "undefined") {
    const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
    if (stored && supportedLocales.some((l) => l.code === stored)) {
      return stored;
    }
  }

  // Check browser language
  const browserLocale = getLocaleFromNavigator();
  if (browserLocale) {
    // Exact match
    if (supportedLocales.some((l) => l.code === browserLocale)) {
      return browserLocale;
    }
    // Language prefix match (e.g., zh-TW -> zh-CN)
    const prefix = browserLocale.split("-")[0];
    const match = supportedLocales.find((l) => l.code.startsWith(prefix));
    if (match) {
      return match.code;
    }
  }

  // Default to English
  return "en";
}

/**
 * Initialize i18n. Call this before mounting the app.
 */
export function initI18n() {
  init({
    fallbackLocale: "en",
    initialLocale: getInitialLocale(),
  });
}

/**
 * Set locale and persist to localStorage.
 */
export function setLocale(newLocale: LocaleCode) {
  locale.set(newLocale);
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(LOCALE_STORAGE_KEY, newLocale);
  }
}

// Export stores for use in components
export { locale, _ };
