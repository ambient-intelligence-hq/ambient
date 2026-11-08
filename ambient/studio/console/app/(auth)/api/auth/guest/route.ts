import { NextResponse } from "next/server";
import { getToken } from "next-auth/jwt";
import { signIn } from "@/app/(auth)/auth";
import { isClosedAccess, isSecureRequest } from "@/lib/constants";

export async function GET(request: Request) {
  if (isClosedAccess()) {
    // Relative: in the standalone server, request.url holds the bind address
    // (0.0.0.0:3000), not the host the browser used.
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    return new Response(null, {
      headers: { Location: `${base}/login` },
      status: 307,
    });
  }

  const { searchParams } = new URL(request.url);
  const rawRedirect = searchParams.get("redirectUrl") || "/";
  const redirectUrl =
    rawRedirect.startsWith("/") && !rawRedirect.startsWith("//")
      ? rawRedirect
      : "/";

  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
    secureCookie: isSecureRequest(request),
  });

  if (token) {
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    return NextResponse.redirect(new URL(`${base}/`, request.url));
  }

  return signIn("guest", { redirect: true, redirectTo: redirectUrl });
}
