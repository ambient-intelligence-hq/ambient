// Unit tests for who the proxy lets through in open and closed access.
// Run: pnpm exec tsx --test proxy.test.ts
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { NextRequest } from "next/server";
import { encode } from "next-auth/jwt";
import { proxy } from "./proxy";

process.env.AUTH_SECRET = "test-secret-for-proxy-tests";

const COOKIE = "authjs.session-token";

async function request(path: string, email?: string) {
  const headers = new Headers();
  if (email) {
    const token = await encode({
      salt: COOKIE,
      secret: process.env.AUTH_SECRET as string,
      token: { email, id: "user-id", type: "regular" },
    });
    headers.set("cookie", `${COOKIE}=${token}`);
  }
  return new NextRequest(`http://localhost:3000${path}`, { headers });
}

// A pass-through response carries this header instead of a redirect or a body.
const passedThrough = (res: Response) =>
  res.headers.get("x-middleware-next") === "1";

afterEach(() => {
  delete process.env.STUDIO_ACCESS;
});

describe("closed access", () => {
  const close = () => {
    process.env.STUDIO_ACCESS = "closed";
  };

  it("sends signed-out pages to /login", async () => {
    close();
    const res = await proxy(await request("/"));
    assert.equal(res.status, 307);
    assert.equal(new URL(res.headers.get("location") ?? "").pathname, "/login");
  });

  it("shows /login but not /register when signed out", async () => {
    close();
    assert.ok(passedThrough(await proxy(await request("/login"))));
    const res = await proxy(await request("/register"));
    assert.equal(new URL(res.headers.get("location") ?? "").pathname, "/login");
  });

  it("rejects signed-out API calls with 401", async () => {
    close();
    const res = await proxy(await request("/api/ambient/files/abc/content"));
    assert.equal(res.status, 401);
  });

  it("treats an old guest cookie as signed out", async () => {
    close();
    const page = await proxy(await request("/", "guest-123"));
    assert.equal(
      new URL(page.headers.get("location") ?? "").pathname,
      "/login"
    );
    const api = await proxy(await request("/api/history", "guest-123"));
    assert.equal(api.status, 401);
  });

  it("lets a signed-in user through", async () => {
    close();
    const res = await proxy(await request("/", "admin@example.com"));
    assert.ok(passedThrough(res));
  });

  it("keeps the health check public", async () => {
    close();
    assert.equal((await proxy(await request("/ping"))).status, 200);
  });
});

describe("open access", () => {
  it("still gives signed-out visitors a guest session", async () => {
    const res = await proxy(await request("/chat/abc"));
    const location = new URL(res.headers.get("location") ?? "");
    assert.equal(location.pathname, "/api/auth/guest");
  });

  it("lets guests through", async () => {
    assert.ok(passedThrough(await proxy(await request("/", "guest-123"))));
  });
});
