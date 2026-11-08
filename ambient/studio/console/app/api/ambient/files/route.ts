import { NextResponse } from "next/server";
import { auth } from "@/app/(auth)/auth";
import { listFiles, uploadFileStream } from "@/lib/ambient/engine";
import { isClosedAccess } from "@/lib/constants";

// This path is excluded from the proxy (see POST below), so both handlers have to
// check sign-in themselves, including a guest cookie left from before closing.
async function isSignedIn() {
  const session = await auth();
  if (!session?.user) {
    return false;
  }
  return !(isClosedAccess() && session.user.type === "guest");
}

// GET /api/ambient/files — list videos in the engine.
export async function GET() {
  if (!(await isSignedIn())) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  try {
    const data = await listFiles();
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}

// POST /api/ambient/files — upload a video (multipart) to the engine.
// Excluded from the proxy (see proxy.ts): the proxy would buffer the body in
// memory and cut it off at 10MB. So sign-in is checked here, and the body is
// streamed through to the engine untouched.
export async function POST(request: Request) {
  if (!(await isSignedIn())) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.startsWith("multipart/form-data") || !request.body) {
    return NextResponse.json(
      { error: "multipart file upload required" },
      { status: 400 }
    );
  }
  try {
    const meta = await uploadFileStream(
      request.body,
      contentType,
      request.signal
    );
    return NextResponse.json(meta);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 502 });
  }
}
