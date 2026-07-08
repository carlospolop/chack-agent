"""Quick smoke-test for the copilot CLI backend."""
from chack_agent.backends.copilot_cli_backend import CopilotCliExecutor

executor = CopilotCliExecutor(
    conversation=[],
    memory_max_messages=10,
    memory_reset_to_messages=5,
    base_system_prompt="You are a helpful assistant. Answer concisely.",
    model_name="gpt-4.1",
    max_turns=5,
    copilot_cli_path="copilot",
    copilot_github_token="",
    tools_config_json="{}",
    allowed_tools_json="[]",
    serialized_tools_override_b64="",
    serialized_tools_append_b64="",
    model_provider="copilot",
    default_model="gpt-4.1",
    social_network_model="gpt-4.1",
    scientific_model="gpt-4.1",
    websearcher_model="gpt-4.1",
    business_model="gpt-4.1",
    product_model="gpt-4.1",
    cli_model="gpt-4.1",
    subchack_model="gpt-4.1",
    social_network_max_turns=5,
    scientific_max_turns=5,
    websearcher_max_turns=5,
    business_max_turns=5,
    product_max_turns=5,
    cli_max_turns=5,
    subchack_max_turns=5,
    min_tools_used=0,
    max_tools_used=0,
    require_task_steps_manager_init_first=False,
    output_schema_json="",
)

result = executor.invoke({"input": "What is 2+2? Answer with just the number."})
print("OUTPUT:", result["output"][:200])
print("STEPS:", len(result["intermediate_steps"]))
print("SUCCESS" if result["output"] and "ERROR" not in result["output"] else "FAILED")
