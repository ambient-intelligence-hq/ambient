// "Start a new chat" from anywhere (sidebar, logo, /new command).
//
// Navigating to "/" alone isn't enough: when you're already on a new chat the
// URL doesn't change, so the chat — and the video already picked for it — would
// carry over. This event tells ActiveChatProvider to mint a fresh chat id, which
// every new chat needs in order to start empty. (The sidebar sits outside that
// provider, hence an event rather than a context call.)
export const NEW_CHAT_EVENT = "ambient:new-chat";

export function startNewChat(router: { push: (href: string) => void }) {
  window.dispatchEvent(new Event(NEW_CHAT_EVENT));
  router.push("/");
}
