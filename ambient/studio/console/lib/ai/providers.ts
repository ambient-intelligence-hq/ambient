import { customProvider, gateway } from "ai";
import { isTestEnvironment } from "../constants";
import { createAmbientModel } from "./ambient-provider";
import { titleModel } from "./models";

export const myProvider = isTestEnvironment
  ? (() => {
      const {
        chatModel,
        titleModel: mockTitleModel,
      } = require("./models.mock");
      return customProvider({
        languageModels: {
          "chat-model": chatModel,
          "title-model": mockTitleModel,
        },
      });
    })()
  : null;

// Studio runs against the Ambient engine, not a raw model gateway: every chat
// model id resolves to the Ambient provider (LanguageModelV2 -> engine). The
// title model uses the gateway only when a gateway key is set; else Ambient.
export function getLanguageModel(modelId: string) {
  if (isTestEnvironment && myProvider) {
    return myProvider.languageModel(modelId);
  }
  return createAmbientModel(modelId);
}

export function getTitleModel() {
  if (isTestEnvironment && myProvider) {
    return myProvider.languageModel("title-model");
  }
  return process.env.AI_GATEWAY_API_KEY
    ? gateway.languageModel(titleModel.id)
    : createAmbientModel("title-model");
}
