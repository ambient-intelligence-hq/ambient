"""stream_chat_completion must surface an in-stream `error` chunk.

Some OpenAI-compatible gateways (vLLM/SGLang deployments, OpenRouter mid-stream)
reject a request with HTTP 200 and a single `data: {"error": {...}}` chunk. That
used to be skipped (no `choices`), so the run ended `end_turn` with an empty
answer and no error event.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from aiohttp import web

from ambient.server.llm_client import stream_chat_completion


def _sse_app(lines: list[str]) -> web.Application:
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for line in lines:
            await resp.write(f"data: {line}\n\n".encode())
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    return app


async def _collect(lines: list[str]) -> list[dict]:
    runner = web.AppRunner(_sse_app(lines))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        return [c async for c in stream_chat_completion(
            model="m", messages=[{"role": "user", "content": "hi"}], tools=[],
            base_url=f"http://127.0.0.1:{port}/v1", api_key="k",
        )]
    finally:
        await runner.cleanup()


class StreamErrorChunkTests(unittest.TestCase):
    def test_error_chunk_raises(self):
        err = {"error": {"message": "The model rejected the request parameters.", "code": "invalid_request"}}
        with self.assertRaisesRegex(RuntimeError, "rejected the request parameters"):
            asyncio.run(_collect([json.dumps(err), "[DONE]"]))

    def test_normal_stream_unaffected(self):
        chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
        got = asyncio.run(_collect([json.dumps(chunk), "[DONE]"]))
        self.assertEqual(got, [chunk])


class OpenAIRequestShapeTests(unittest.TestCase):
    """api.openai.com rejects OpenRouter's fields and max_tokens on newer models."""

    def _body(self, url):
        import ambient.server.llm_client as lc
        seen = {}

        class Resp:
            status = 200
            content = _Lines()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class Session:
            def __init__(self, **_k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, _endpoint, headers=None, json=None):
                seen.update(json)
                return Resp()

        async def go():
            async for _ in lc.stream_chat_completion(model="m", messages=[], tools=[], base_url=url, api_key="k"):
                pass
        with mock.patch.object(lc.aiohttp, "ClientSession", Session):
            asyncio.run(go())
        return seen

    def test_openai_gets_only_its_own_fields(self):
        body = self._body("https://api.openai.com/v1")
        self.assertEqual(body["max_completion_tokens"], 16000)
        for field in ("max_tokens", "usage", "reasoning"):
            self.assertNotIn(field, body)
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_openrouter_is_unchanged(self):
        body = self._body("https://openrouter.ai/api/v1")
        self.assertEqual(body["max_tokens"], 16000)
        self.assertEqual(body["usage"], {"include": True})
        self.assertEqual(body["reasoning"], {"enabled": True})


class _Lines:
    def __aiter__(self):
        async def gen():
            yield b"data: [DONE]\n"
        return gen()


class RetryTests(unittest.TestCase):
    """Transient upstream failures are retried, but only before any output."""

    def _serve(self, responses):
        calls = []

        async def handler(request):
            calls.append(1)
            status, lines = responses[min(len(calls), len(responses)) - 1]
            if status != 200:
                return web.Response(status=status, text='{"error":{"message":"Internal Server Error"}}')
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for line in lines:
                await resp.write(f"data: {line}\n\n".encode())
            await resp.write_eof()
            return resp

        async def go():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                return [c async for c in stream_chat_completion(
                    model="m", messages=[{"role": "user", "content": "hi"}], tools=[],
                    base_url=f"http://127.0.0.1:{port}/v1", api_key="k")]
            finally:
                await runner.cleanup()
        return go, calls

    def setUp(self):
        import ambient.server.llm_client as lc
        self._delays = lc._RETRY_DELAYS
        lc._RETRY_DELAYS = (0.01, 0.01)

    def tearDown(self):
        import ambient.server.llm_client as lc
        lc._RETRY_DELAYS = self._delays

    def test_transient_500_is_retried(self):
        chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
        go, calls = self._serve([(500, []), (200, [json.dumps(chunk), "[DONE]"])])
        self.assertEqual(asyncio.run(go()), [chunk])
        self.assertEqual(len(calls), 2)

    def test_gives_up_after_retries(self):
        go, calls = self._serve([(503, [])])
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            asyncio.run(go())
        self.assertEqual(len(calls), 3)

    def test_client_errors_not_retried(self):
        go, calls = self._serve([(400, [])])
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            asyncio.run(go())
        self.assertEqual(len(calls), 1)

    def test_rejection_chunk_not_retried(self):
        go, calls = self._serve([(200, [json.dumps({"error": {"message": "rejected"}}), "[DONE]"])])
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            asyncio.run(go())
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
