from ambient.tools.search_clip import search_clip, SearchClipTool
from ambient.tools.focus_clip import focus_clip, FocusClipTool
from ambient.tools.annotate_frames import annotate_frames, AnnotateFramesTool
from ambient.tools.grab_frames import grab_frames, GrabFramesTool
from ambient.tools.draw_bounding_box import draw_bounding_box, DrawBoundingBoxTool
from ambient.tools.draw_point import draw_point, DrawPointTool
from ambient.config import get_model_modalities, settings, model_modalities
from ambient.tools.self_focus_clip import self_focus_clip, SelfFocusClipTool


TOOL_REGISTRY = {}

# TOOL_REGISTRY["get_video_description"] = get_video_description
TOOL_REGISTRY["search_clip"] = search_clip
TOOL_REGISTRY["focus_clip"] = focus_clip

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_clip",
            "description": "Search a specific portion of the video for the query. The end time should be within 5 mins from the start_time. An video analysis will analyse the portion of the video and return if the clip is relevant to the query and a answer to the query.",
            "parameters": SearchClipTool.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_clip",
            "description": "Focus a specific portion of the video for the query. The end time should be within 5 mins from the start_time. An video analysis will analyse the portion of the video and return a detailed description of the clip.",
            "parameters": FocusClipTool.model_json_schema(),
        },
    },
]

# override the default tools when self video analysis tool is enabled
if settings.self_video_analysis_tool:
    DEFAULT_TOOLS = [{
        "type": "function",
        "function": {
            "name": "focus_clip",
            "description": "Focus a specific portion of the video for the query. The end time should be within 5 mins from the start_time. You will receive a high fidelity clip of the video between the start and end time for detailed inspection. Carefully review the clip and decide next actions. ",
            "parameters": SelfFocusClipTool.model_json_schema(),
        },
    }]
    TOOL_REGISTRY = {}
    TOOL_REGISTRY["focus_clip"] = self_focus_clip

TEXT_MODALITY_ONLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "annotate_frames",
            "description": "Get the bounding box coordinates for the given annotation description. The start and end time should be within 5 seconds from each other. You should use this only when your final response requires bounding box coordinates.",
            "parameters": AnnotateFramesTool.model_json_schema(),
        },
    }
]

IMAGE_MODALITY_ONLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grab_frames",
            "description": "Self Read the frames from the video between the start and end time for closer inspection. The start and end time should be within 5 seconds from each other.",
            "parameters": GrabFramesTool.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_bounding_box",
            "description": "Draw one or more bounding boxes on the frame at the given timestamp. Pass a list of boxes, each with its own coordinates in the format [y_min, x_min, y_max, x_max] on a 0-1000 normalizedgrid and a label describing it. Use this tool to verify your bounding box predictions. You should call the grab_frames tool before this to inspect the exact frame and its dimention, before drawing the bounding box.",
            "parameters": DrawBoundingBoxTool.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_point",
            "description": "Visualize one or more point / click coordinates on the frame at the given timestamp. Pass a list of points, each as [x, y] on a 0-1000 grid (x horizontal, y vertical) with a label. Returns the annotated frame plus each click point in pixel and normalized coordinates. Use this tool to verify predicted point/click locations before relying on them.",
            "parameters": DrawPointTool.model_json_schema(),
        },
    },
]

TOOLS = DEFAULT_TOOLS

agent_modalities = get_model_modalities(settings.agent_model)

# If the agent model does not support images, add the annotate_frames tool
if agent_modalities and model_modalities.IMAGE not in agent_modalities:
    TOOLS.extend(TEXT_MODALITY_ONLY_TOOLS)
    TOOL_REGISTRY["annotate_frames"] = annotate_frames

# If the agent model supports images, add the grab_frames tool
if agent_modalities and model_modalities.IMAGE in agent_modalities:
    TOOLS.extend(IMAGE_MODALITY_ONLY_TOOLS)
    TOOL_REGISTRY["grab_frames"] = grab_frames
    TOOL_REGISTRY["draw_bounding_box"] = draw_bounding_box
    TOOL_REGISTRY["draw_point"] = draw_point
