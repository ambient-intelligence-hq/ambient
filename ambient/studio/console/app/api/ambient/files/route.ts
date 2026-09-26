import { NextResponse } from "next/server";
import { listFiles, uploadFile } from "@/lib/ambient/engine";

// GET /api/ambient/files — list videos in the engine.
export async function GET() {
  try {
    const data = await listFiles();
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

// POST /api/ambient/files — upload a video (multipart) to the engine.
export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof Blob)) {
      return NextResponse.json({ error: "file part required" }, { status: 400 });
    }
    const name = (file as File).name || "upload.mp4";
    const meta = await uploadFile(file, name);
    return NextResponse.json(meta);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}
