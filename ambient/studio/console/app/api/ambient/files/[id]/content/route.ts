import type { NextRequest } from "next/server";
import { proxyContent } from "@/lib/ambient/engine";

// GET /api/ambient/files/:id/content — proxy the engine's ranged video stream so
// the browser <video> element can play/seek without ever seeing the engine key.
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const range = request.headers.get("range");
  const upstream = await proxyContent(id, range);

  // Pass through status (200/206/416) and the headers a media element needs.
  const headers = new Headers();
  for (const h of ["content-type", "content-length", "content-range", "accept-ranges", "cache-control"]) {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
