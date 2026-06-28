from ambient.tools.video_description import get_video_description, VideoDescriptionTool
from ambient.tools.search_clip import search_clip, SearchClipTool
from ambient.tools.focus_clip import focus_clip, FocusClipTool

TOOL_REGISTRY = {}

# TOOL_REGISTRY["get_video_description"] = get_video_description
TOOL_REGISTRY["search_clip"] = search_clip
TOOL_REGISTRY["focus_clip"] = focus_clip

TOOLS = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_video_description",
    #         "description": "Get the highlevel description of the video",
    #         "parameters": VideoDescriptionTool.model_json_schema(),
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "search_clip",
            "description": "Search a specific portion of the video for the query. The end time should be within 5 mins from the start_time. An video analysis will analyse the portion of the video and return if the clip is relevant to the query and a answer to the query.",
            "parameters": SearchClipTool.model_json_schema(),
        }
    },
    {
        "type": "function",
        "function": {
            "name": "focus_clip",
            "description": "Focus a specific portion of the video for the query. The end time should be within 5 mins from the start_time. An video analysis will analyse the portion of the video and return a detailed description of the clip.",
            "parameters": FocusClipTool.model_json_schema(),
        }
    },
]   