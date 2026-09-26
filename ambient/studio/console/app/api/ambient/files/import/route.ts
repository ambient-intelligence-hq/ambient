import { NextResponse } from "next/server";
import { importYouTube } from "@/lib/ambient/engine";

// POST /api/ambient/files/import — import a video from a YouTube URL via the
// engine (keeps the engine key server-side).
export async function POST(request: Request) {
  try {
    const { url } = await request.json();
    if (typeof url !== "string" || !url.trim()) {
      return NextResponse.json({ error: "url required" }, { status: 400 });
    }
    const meta = await importYouTube(url.trim());
    return NextResponse.json(meta);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}
