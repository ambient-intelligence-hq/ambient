import { type NextRequest, NextResponse } from "next/server";
import { getToken } from "next-auth/jwt";
import { guestRegex, isClosedAccess, isSecureRequest } from "./lib/constants";

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (pathname.startsWith("/ping")) {
    return new Response("pong", { status: 200 });
  }

  if (pathname.startsWith("/api/auth")) {
    return NextResponse.next();
  }

  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
    secureCookie: isSecureRequest(request),
  });

  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

  // A guest cookie from before the install was closed doesn't count as signed in.
  if (isClosedAccess() && (!token || guestRegex.test(token.email ?? ""))) {
    if (pathname === "/login") {
      return NextResponse.next();
    }
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    }
    return NextResponse.redirect(new URL(`${base}/login`, request.url));
  }

  if (!token) {
    const redirectUrl = encodeURIComponent(new URL(request.url).pathname);

    return NextResponse.redirect(
      new URL(`${base}/api/auth/guest?redirectUrl=${redirectUrl}`, request.url)
    );
  }

  const isGuest = guestRegex.test(token?.email ?? "");

  if (token && !isGuest && ["/login", "/register"].includes(pathname)) {
    return NextResponse.redirect(new URL(`${base}/`, request.url));
  }

  return NextResponse.next();
}

export const config = {
  // `api/ambient/files$` (the video upload) is left out on purpose: a matched
  // request's body is buffered in memory and truncated at 10MB before the route
  // sees it. That route checks sign-in itself. Its subpaths stay matched.
  // The favicon and brand mark are public: the login page shows them signed out.
  matcher: [
    "/",
    "/chat/:id",
    "/login",
    "/register",

    "/((?!_next/static|_next/image|icon.png|ambient-mark.jpg|sitemap.xml|robots.txt|api/ambient/files$).*)",
  ],
};
