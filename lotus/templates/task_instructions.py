import re
from typing import Any

import pandas as pd

import lotus
from lotus.dtype_extensions import ImageDtype
from lotus.types import ReasoningStrategy, SerializationFormat


def cot_formatter(reasoning, answer):
    return f"""Reasoning:\n{reasoning}\n\nAnswer: {answer}"""


def answer_only_formatter(answer):
    return f"""Answer: {answer}"""


def deepseek_cot_formatter():
    return """Please think through your reasoning step by step, then provide your final answer.
    You must put your reasoning inside the <think></think> tags, then provide your 
    final answer after the </think> tag with the format: Answer: your answer."""


def cot_prompt_formatter(reasoning_instructions: str = "", answer_instructions: str = "") -> str:
    reasoning_instructions = f"<Your reasoning here. {reasoning_instructions}>"
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Let's think step by step. Use the following format to provide your answer:
        {cot_formatter(reasoning_instructions, answer_instructions)}
        """


def non_cot_prompt_formatter(answer_instructions: str = "") -> str:
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Use the following format to provide your answer:
            {answer_only_formatter(answer_instructions)}
            """


def context_formatter(
    multimodal_data: dict[str, Any] | str,
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(multimodal_data, str):
        text = multimodal_data
        image_inputs: list[dict[str, str]] = []
    elif isinstance(multimodal_data, dict):
        image_data: dict[str, str] = multimodal_data.get("image", {})
        _image_inputs: list[tuple[dict, dict]] = [
            (
                {
                    "type": "text",
                    "text": f"[{key.capitalize()}]: \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            )
            for key, base64_image in image_data.items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        text = multimodal_data["text"] or ""
    else:
        raise ValueError("multimodal_data must be a dictionary or a string")
    return text, image_inputs


def user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction_with_tag: str | None = None,
) -> dict[str, Any]:
    text, image_inputs = context_formatter(multimodal_data)
    if not image_inputs or len(image_inputs) == 0:
        return {
            "role": "user",
            "content": f"Context:\n{text}\n\n{user_instruction_with_tag}",
        }
    content = [{"type": "text", "text": f"Context:\n{text}"}] + image_inputs
    if user_instruction_with_tag:
        content.append({"type": "text", "text": f"\n\n{user_instruction_with_tag}"})
    return {
        "role": "user",
        "content": content,
    }


def filter_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[bool] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | None = None,
    reasoning_instructions: str = "",
    system_prompt: str | None = None,
    output_tokens: tuple[str, str] = ("True", "False"),
) -> list[dict[str, str]]:
    positive_token, negative_token = output_tokens
    answer_instructions = f"The answer should be either {positive_token} or {negative_token}"

    default_sys_instruction = """The user will provide a claim and some relevant context.
    Your job is to determine whether the claim is true for the given context.
     """

    sys_instruction = system_prompt or default_sys_instruction
    if strategy == ReasoningStrategy.COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    elif not system_prompt:
        sys_instruction += non_cot_prompt_formatter(answer_instructions=answer_instructions)

    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        assert isinstance(examples_multimodal_data, list) and isinstance(examples_answer, list)
        assert len(examples_multimodal_data) == len(examples_answer)

        if cot_reasoning:
            # users don't have to provide cot reasoning examples
            # but if they do, the number of examples must match
            assert isinstance(cot_reasoning, list)
            assert len(examples_multimodal_data) == len(examples_answer) == len(cot_reasoning)

        for idx in range(len(examples_multimodal_data)):
            ex_multimodal_data = examples_multimodal_data[idx]
            ex_ans = examples_answer[idx]
            if isinstance(ex_ans, str):
                ex_ans_token = ex_ans.lower() == positive_token.lower()
            elif isinstance(ex_ans, bool):
                ex_ans_token = positive_token if ex_ans else negative_token
            else:
                raise ValueError(f"Invalid example answer type: {type(ex_ans)}")
            content = ""

            if cot_reasoning:
                content = cot_formatter(cot_reasoning[idx], ex_ans_token)
            elif strategy == ReasoningStrategy.COT:
                content = cot_formatter("Reasoning omitted", ex_ans_token)
            else:
                content = answer_only_formatter(ex_ans_token)

            messages.extend(
                [
                    user_message_formatter(ex_multimodal_data, f"Claim: {user_instruction}"),
                    {
                        "role": "assistant",
                        "content": content,
                    },
                ]
            )
    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_instruction = f"Claim: {user_instruction}\n\n{deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_instruction))
    else:
        messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


def map_formatter_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]],
    examples_answer: list[str],
    cot_reasoning: list[str],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        "You must give your reasoning and then your final answer"
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    for idx in range(len(examples_multimodal_data)):
        ex_df_txt = examples_multimodal_data[idx]
        ex_ans = examples_answer[idx]
        cot = cot_reasoning[idx]
        messages.extend(
            [
                user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                {
                    "role": "assistant",
                    "content": f"Reasoning:\n{cot}\n\nAnswer: {ex_ans}",
                },
            ]
        )

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter_zs_cot(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        'First give your reasoning. Then you MUST end your output with "Answer: your answer"'
    )
    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def map_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    user_instruction: str,
    examples_multimodal_data: list[dict[str, Any]] | None = None,
    examples_answer: list[str] | None = None,
    cot_reasoning: list[str] | None = None,
    strategy: ReasoningStrategy | str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    sys_instruction = system_prompt or (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    if cot_reasoning:
        assert examples_multimodal_data is not None and examples_answer is not None
        return map_formatter_cot(
            multimodal_data, user_instruction, examples_multimodal_data, examples_answer, cot_reasoning, system_prompt
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        return map_formatter_zs_cot(multimodal_data, user_instruction, system_prompt)

    messages = [
        {"role": "system", "content": sys_instruction},
    ]

    if examples_multimodal_data:
        assert examples_answer is not None
        for ex_df_txt, ex_ans in zip(examples_multimodal_data, examples_answer):
            messages.extend(
                [
                    user_message_formatter(ex_df_txt, f"Instruction: {user_instruction}"),
                    {"role": "assistant", "content": str(ex_ans)},
                ]
            )

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_intructions = f"Instruction: {user_instruction}\n\n{deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_intructions))
    else:
        messages.append(user_message_formatter(multimodal_data, f"Instruction: {user_instruction}"))
    return messages


def extract_formatter(
    model: lotus.models.LM,
    multimodal_data: dict[str, Any],
    output_cols: dict[str, str | None],
    extract_quotes: bool = True,
    strategy: ReasoningStrategy | None = None,
) -> list[dict[str, str]]:
    output_col_names = list(output_cols.keys())
    # Set the description to be the key if no value is provided
    output_cols_with_desc: dict[str, str] = {col: col if desc is None else desc for col, desc in output_cols.items()}

    all_fields = output_col_names
    quote_fields = []
    if extract_quotes:
        quote_fields = [f"{col}_quote" for col in output_col_names]
        all_fields += quote_fields

    fields_str = ", ".join(all_fields)

    # Add CoT reasoning instructions to the system prompt if needed
    if strategy == ReasoningStrategy.COT:
        reasoning_instructions = "Think through each extraction step by step."
        answer_instructions = f"Provide the JSON response with fields: {fields_str}"
        cot_instruction = cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    elif strategy == ReasoningStrategy.ZS_COT:
        reasoning_instructions = "Think through each extraction step by step."
        answer_instructions = f"Provide the JSON response with fields: {fields_str}"
        cot_instruction = cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    else:
        cot_instruction = ""

    if extract_quotes:
        sys_instruction = (
            "The user will provide the columns that need to be extracted and some relevant context.\n"
            f"Your job is to extract these columns and provide only a concise value for each field "
            f"and the corresponding full quote for each field in the '{', '.join(quote_fields)}' fields.\n"
            f"Here is a description of each field: {output_cols_with_desc}\n"
            f"The response should be valid JSON format with the following fields: {fields_str}.\n"
        )
    else:
        sys_instruction = (
            "The user will provide the columns that need to be extracted and some relevant context.\n"
            f"Your job is to extract these columns and provide only a concise value for each field.\n"
            f"Here is a description of each field: {output_cols_with_desc}\n"
            f"The response should be valid JSON format with the following fields: {fields_str}.\n"
        )

    # Add CoT instruction to system prompt if applicable
    if cot_instruction:
        sys_instruction += "\n" + cot_instruction

    messages = [
        {"role": "system", "content": sys_instruction},
        user_message_formatter(multimodal_data),
    ]

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        user_intructions = f"Instruction: {deepseek_cot_formatter()}"
        messages.append(user_message_formatter(multimodal_data, user_intructions))

    return messages


# returns a list of strings corresponding to df rows
def df2text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Formats the given DataFrame into a string containing info from cols."""

    def custom_format_row(x: pd.Series, cols: list[str]) -> str:
        return "".join([f"[{cols[i].capitalize()}]: «{x[cols[i]]}»\n" for i in range(len(cols))])

    def clean_and_escape_column_name(column_name: str) -> str:
        clean_name = re.sub(r"[^\w]", "", column_name)  # Remove spaces and special characters
        return clean_name

    # take cols that are in df
    cols = [col for col in cols if col in df.columns]
    if len(cols) == 0:
        return [""] * len(df)

    projected_df = df[cols]
    formatted_rows: list[str] = []

    if lotus.settings.serialization_format == SerializationFormat.DEFAULT:
        formatted_rows = projected_df.apply(lambda x: custom_format_row(x, cols), axis=1).tolist()
    elif lotus.settings.serialization_format == SerializationFormat.JSON:
        formatted_rows = projected_df.to_json(orient="records", lines=True).splitlines()
    elif lotus.settings.serialization_format == SerializationFormat.XML:
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError(
                "The 'lxml' library is required for XML serialization. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[xml]'"
            )
        projected_df = projected_df.rename(columns=lambda x: clean_and_escape_column_name(x))
        full_xml = projected_df.to_xml(root_name="data", row_name="row", pretty_print=False, index=False)
        root = ET.fromstring(full_xml)
        formatted_rows = [ET.tostring(row, encoding="unicode", method="xml") for row in root.findall("row")]

    return formatted_rows


def lineage_formatter(
    model: lotus.models.LM,
    claim_text: str,
    events_blob: str,
    instruction: str | None = None,
    max_evidence: int = 5,
    strategy: ReasoningStrategy | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for claim→event provenance selection.

    The model must return JSON with:
      - supported: bool
      - evidence_event_ids: list[str] (event_id values from the numbered event list)
      - lineage_path: ordered list of event_ids or tool names explaining the support chain
      - rationale: short explanation
    """
    task = instruction or (
        "Decide whether the claim is supported by the agent event stream. "
        "If supported, select the minimal set of evidence events (by event_id) "
        "that justify the claim, in causal order."
    )
    answer_schema = (
        '{"supported": true|false, '
        f'"evidence_event_ids": ["id1", ...] (at most {max_evidence}), '
        '"lineage_path": ["id_or_tool", ...], '
        '"rationale": "brief explanation"}'
    )

    if strategy in (ReasoningStrategy.COT, ReasoningStrategy.ZS_COT):
        cot_instruction = cot_prompt_formatter(
            reasoning_instructions="Check each event for evidence that entails or refutes the claim.",
            answer_instructions=f"Provide valid JSON: {answer_schema}",
        )
        sys_instruction = (
            "You are a claim-provenance judge over agent event streams.\n"
            f"Task: {task}\n"
            "Only use event_id values that appear in the event list.\n"
            f"{cot_instruction}"
        )
    else:
        sys_instruction = (
            "You are a claim-provenance judge over agent event streams.\n"
            f"Task: {task}\n"
            "Only use event_id values that appear in the event list.\n"
            f"Respond with valid JSON only: {answer_schema}"
        )

    user_content = f"[Claim]: «{claim_text}»\n\n[Events]:\n{events_blob}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_content},
    ]

    if strategy == ReasoningStrategy.ZS_COT and model.is_deepseek():
        messages.append({"role": "user", "content": f"Instruction: {deepseek_cot_formatter()}"})

    return messages


def df2multimodal_info(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """
    Formats the given DataFrame into a string containing info from cols.
    Return a list of dictionaries, each containing text and image data.
    """
    image_cols = [col for col in cols if isinstance(df[col].dtype, ImageDtype)]
    text_cols = [col for col in cols if col not in image_cols]
    text_rows = df2text(df, text_cols)
    multimodal_data = [
        {
            "text": text_rows[i],
            "image": {col.capitalize(): df[col].array.get_image(i, "base64") for col in image_cols},
        }
        for i in range(len(df))
    ]
    return multimodal_data


def merge_multimodal_info(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges two multimodal info lists into one. Each row of first is merged with each row of second.

    Args:
        first: list of multimodal info dictionaries
        second: list of multimodal info dictionaries

    Returns:
        list of merged multimodal info dictionaries
    """
    return [
        {
            "text": f"{first[i]['text']}\n{second[j]['text']}"
            if first[i]["text"] != "" and second[j]["text"] != ""
            else first[i]["text"] + second[j]["text"],
            "image": {**first[i]["image"], **second[j]["image"]},
        }
        for i in range(len(first))
        for j in range(len(second))
    ]


def li2text(li: list[str], name: str) -> str:
    return "".join([f"[{name}] {li[i]}\n" for i in range(len(li))])
