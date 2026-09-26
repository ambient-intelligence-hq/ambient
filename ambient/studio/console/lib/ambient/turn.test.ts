// Unit tests for the turn persistence/resume decisions (lib/ambient/turn.ts).
// Run: pnpm exec tsx --test lib/ambient/turn.test.ts
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  isRunComplete,
  planTurnAssistantSave,
  prepareMessagesForLoad,
} from "./turn";

const user = (id: string) => ({ id, role: "user", metadata: null });
const done = (id: string) => ({ id, role: "assistant", metadata: { runComplete: true } });
const partial = (id: string) => ({ id, role: "assistant", metadata: null });

describe("isRunComplete", () => {
  it("is true only for an explicit runComplete marker", () => {
    assert.equal(isRunComplete({ runComplete: true }), true);
    assert.equal(isRunComplete({ runComplete: false }), false);
    assert.equal(isRunComplete({ usage: {} }), false);
    assert.equal(isRunComplete(null), false);
    assert.equal(isRunComplete(undefined), false);
  });
});

describe("planTurnAssistantSave", () => {
  it("inserts the first answer of a turn", () => {
    const plan = planTurnAssistantSave([user("u1")], done("a1"));
    assert.deepEqual(plan, { skip: false, deleteIds: [], exists: false });
  });

  it("replaces a partial left by a cut stream with the resumed answer", () => {
    // Reload mid-run: the cut stream saved partial p; the resume (new id) finishes.
    const plan = planTurnAssistantSave([user("u1"), partial("p")], done("r"));
    assert.deepEqual(plan, { skip: false, deleteIds: ["p"], exists: false });
  });

  it("never clobbers a complete answer with a later partial", () => {
    // The resume finished first; a cut stream saves its partial afterwards.
    const plan = planTurnAssistantSave([user("u1"), done("r")], partial("p"));
    assert.deepEqual(plan, { skip: true });
  });

  it("updates in place when the same message is saved again", () => {
    const plan = planTurnAssistantSave([user("u1"), partial("a")], done("a"));
    assert.deepEqual(plan, { skip: false, deleteIds: [], exists: true });
  });

  it("only touches the latest turn, never an earlier turn's answer", () => {
    // The old resume saved into "the last assistant row" — the previous turn's
    // answer — when the current turn had none yet.
    const rows = [user("u1"), done("a1"), user("u2")];
    const plan = planTurnAssistantSave(rows, done("a2"));
    assert.deepEqual(plan, { skip: false, deleteIds: [], exists: false });
  });

  it("collapses duplicate rows of the same turn", () => {
    const rows = [user("u1"), partial("p1"), partial("p2")];
    const plan = planTurnAssistantSave(rows, done("r"));
    assert.deepEqual(plan, { skip: false, deleteIds: ["p1", "p2"], exists: false });
  });
});

describe("prepareMessagesForLoad", () => {
  const sent = (messageId: string) => ({ messageId, state: "sent" });

  it("drops a partial answer so the client resumes the running turn", () => {
    const msgs = [user("u1"), done("a1"), user("u2"), partial("p")];
    const out = prepareMessagesForLoad(msgs, sent("u2"));
    assert.equal(out.resumable, true);
    assert.deepEqual(out.messages.map((m) => m.id), ["u1", "a1", "u2"]);
  });

  it("resumes a turn whose answer was never saved", () => {
    const out = prepareMessagesForLoad([user("u1")], sent("u1"));
    assert.equal(out.resumable, true);
    assert.deepEqual(out.messages.map((m) => m.id), ["u1"]);
  });

  it("leaves a completed turn alone", () => {
    const msgs = [user("u1"), done("a1")];
    const out = prepareMessagesForLoad(msgs, sent("u1"));
    assert.equal(out.resumable, false);
    assert.equal(out.messages, msgs);
  });

  it("waits out a pending send (the run will exist momentarily)", () => {
    const out = prepareMessagesForLoad([user("u1")], { messageId: "u1", state: "pending" });
    assert.equal(out.resumable, true);
  });

  it("never resumes a turn the engine rejected (e.g. 409 run in flight)", () => {
    // Replaying the engine's latest run here would show the *previous* turn.
    const msgs = [user("u1"), done("a1"), user("u2"), partial("err")];
    const out = prepareMessagesForLoad(msgs, { messageId: "u2", state: "failed" });
    assert.equal(out.resumable, false);
    assert.equal(out.messages, msgs);
  });

  it("never resumes without an anchor for the latest user message", () => {
    // Anchor belongs to an older turn (or is absent, e.g. pre-fix chats).
    const msgs = [user("u1"), done("a1"), user("u2"), partial("p")];
    assert.equal(prepareMessagesForLoad(msgs, sent("u1")).resumable, false);
    assert.equal(prepareMessagesForLoad(msgs, null).resumable, false);
  });
});
