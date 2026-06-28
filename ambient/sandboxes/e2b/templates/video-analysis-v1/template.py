from e2b import AsyncTemplate

template = (
    AsyncTemplate()
    .from_dockerfile("Dockerfile")
    .run_cmd("echo Hello World E2B!")
)