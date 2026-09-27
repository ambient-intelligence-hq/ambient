import { generateDummyPassword } from "./db/utils";

export const isProductionEnvironment = process.env.NODE_ENV === "production";
export const isDevelopmentEnvironment = process.env.NODE_ENV === "development";
export const isTestEnvironment = Boolean(
  process.env.PLAYWRIGHT_TEST_BASE_URL ||
    process.env.PLAYWRIGHT ||
    process.env.CI_PLAYWRIGHT
);

export const guestRegex = /^guest-\d+$/;

// Whether the session cookie for this request is the HTTPS-only
// (`__Secure-`) variant. Must match Auth.js's own rule — it issues the secure
// cookie only when the request URL is https (behind a TLS proxy, per
// X-Forwarded-Proto). Keying this off NODE_ENV instead (the upstream template)
// breaks a production build served over plain http, e.g. a self-hosted
// http://localhost:3000: the cookie is set under one name and read under the
// other, so every request bounces to guest sign-in in an endless loop.
export function isSecureRequest(request: Request): boolean {
  const forwarded = request.headers.get("x-forwarded-proto");
  const proto = forwarded
    ? `${forwarded.split(",")[0].trim()}:`
    : new URL(request.url).protocol;
  return proto === "https:";
}

export const DUMMY_PASSWORD = generateDummyPassword();

export const suggestions = [
  "Summarize this video.",
  "What happens in the first minute?",
  "List the key moments with timestamps.",
  "Describe the main people and objects on screen.",
];
