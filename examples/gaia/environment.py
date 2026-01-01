"""GAIA environment implementation."""

from __future__ import annotations

import os
from contextlib import redirect_stderr
from dataclasses import dataclass
from typing import Any, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from timestep.environment import OpenAIEnvironment


@dataclass
class GAIAConfig:
    """Configuration for GAIA environment."""

    repo_id: str = "gaia-benchmark/GAIA"
    # Example configs mentioned in the dataset card include things like "2023_level1".
    # You can also load other levels/splits as exposed by the dataset.
    config_name: str = "2023_level1"
    split: str = "validation"  # "validation" for public validation, "test" for hidden answers
    hf_token_env: str = "HF_TOKEN"  # set this in your shell


class GAIAEnvironment(OpenAIEnvironment):
    """
    One GAIA question per episode.

    Observation: Dict with "messages" key containing list of OpenAI messages
    Action: Dict with "messages" key containing list with assistant message
    Reward: 1.0 if exact match (when reference answer available), else 0.0.
    Done: True after a final answer action (assistant message with content, no tool_calls).
    """

    def __init__(self, cfg: GAIAConfig, use_reflection: bool = False):
        """
        Initialize GAIA environment.

        Args:
            cfg: GAIA configuration
            use_reflection: If True, use multi-agent reflection (draft, reflection, revision)
        """
        super().__init__()
        self.cfg = cfg
        self.use_reflection = use_reflection
        self._data_dir: Optional[str] = None
        self._ds: Optional[Any] = None  # datasets.Dataset (optional dependency)
        self._idx: int = -1
        self._current: Optional[dict[str, Any]] = None
        self._draft_string: Optional[str] = None
        self._reflection_string: Optional[str] = None
        self._original_question: Optional[str] = None
        self._draft_retry_count: int = 0
        self.max_draft_retries: int = 2

        # Set up agents based on use_reflection flag
        if use_reflection:
            self._agents = ["draft", "reflection", "decision", "revision"]
        else:
            self._agents = ["draft"]

        # Initialize system prompts for each agent
        self._agent_system_prompts = {
            "draft": """You are an expert research assistant solving a single benchmark question.
Your task is to produce a draft answer that is as accurate and well-supported as possible.

CRITICAL: You MUST provide a specific answer to the question, even if the problem is ambiguous or you cannot fully compute it. Do not refuse to answer - always provide the best answer you can determine, even if it requires making reasonable assumptions.

Guidelines:
- Use external tools if necessary to verify facts or resolve uncertainty.
- State assumptions explicitly when needed.
- Resolve ambiguities logically rather than guessing.
- If the problem is ambiguous, commit to the most reasonable interpretation and provide an answer based on that interpretation. If multiple interpretations are equally reasonable, provide conditional answers for each interpretation.
- Include explanations or intermediate reasoning if they help ensure correctness.
- Do not attempt to optimize for brevity.
- Treat this output as a working draft, not a final answer.""",
            "reflection": """You are a critical reviewer evaluating a draft answer to a benchmark question.
Your goal is to improve both correctness and succinctness in the eventual final answer.

CRITICAL: First and foremost, verify whether the draft's final answer/conclusion is correct. An incorrect final answer is the most serious correctness issue, even if the reasoning appears sound.

IMPORTANT: Plausible reasoning does not guarantee a correct conclusion. For mathematical, logical, or riddle questions, a conclusion is only correct if it is definitively proven. If the reasoning has gaps, makes unproven assumptions, or relies on intuition rather than rigorous proof, flag the conclusion as uncertain or likely incorrect, even if the reasoning seems plausible.

VERIFICATION REQUIREMENT: For mathematical, logical, or riddle questions, you must independently verify the conclusion is correct. Do not validate a conclusion as "correct" just because the reasoning seems sound - the conclusion itself must be verified independently of the reasoning quality. Correct reasoning does not guarantee a correct conclusion. If you cannot definitively verify the conclusion is correct, flag it as uncertain or likely incorrect rather than validating it.

Evaluate the draft along two axes:

1. Substance (correctness)
- Is the draft's final answer/conclusion correct? This is the highest priority.
- For mathematical, logical, or riddle questions: the conclusion must be definitively proven, not just plausibly argued. If the reasoning relies on intuition, unproven assumptions, or has logical gaps, flag this as a critical correctness issue.
- Do not validate a conclusion as "correct" based solely on plausible reasoning - require definitive proof or evidence.
- If you cannot definitively verify the conclusion is correct, flag it as uncertain or likely incorrect rather than validating it.
- When evaluating counts or lists, verify completeness: ensure all items in the specified range are accounted for
- Be skeptical of conclusions labeled as "likely correct" - if not definitively proven, flag as a correctness risk
- For questions asking "how many", verify the count matches what the question asks for (e.g., if asking for items in a range, ensure the range boundaries are correctly interpreted)
- Factual errors or questionable claims
- Logical errors in reasoning that lead to wrong conclusions
- Missing assumptions or unresolved ambiguities
- Reasoning gaps or unjustified conclusions
- Be especially skeptical of conclusions that seem counterintuitive or when the reasoning, while plausible, might lead to the wrong answer

2. Excess (verbosity)
- Explanations that do not affect the final answer
- Redundant or obvious statements
- Overly cautious, defensive, or verbose phrasing
- Details appropriate for reasoning but unnecessary in a final response

Rules:
- Do not rewrite the answer.
- Do not polish language for style alone.
- Be concrete: reference specific parts that can be fixed, shortened, or removed.
- Prefer actionable guidance such as "can be deleted" or "can be reduced to X".
- If the draft's conclusion is incorrect, explicitly state this as a critical correctness issue.
- If the draft is already strong, still look for opportunities to compress.""",
            "decision": """You are a decision agent. Based on the reflection critique, determine if the draft answer has critical correctness issues that require the draft agent to retry, or if the draft is acceptable and can proceed to revision.

Your task is to evaluate the reflection critique and decide whether the draft needs to be corrected by the draft agent before proceeding to revision.

Output only "retry" if the reflection critique identifies critical correctness issues that would require the draft agent to re-analyze the question.
Output only "proceed_to_revision" if the draft is acceptable (even if it has minor issues that can be fixed during revision).""",
            "revision": """You are a revision-focused assistant.
Your task is to produce the most succinct correct answer to the original question, using the reviewer's critique.

CRITICAL RULES (apply to ALL questions, in priority order):
1. CORRECTNESS FIRST: The critique identifies correctness issues that MUST be fixed. Read the critique carefully and address ALL correctness problems it identifies before making the answer shorter.
2. UNDERSTAND UNITS AND SCALES: Carefully read the question's unit requirements. The question may ask for a quantity in a specific scale (e.g., "how many thousand X", "how many million Y"). When this happens, you must convert your computed value to match that scale by dividing by the appropriate factor. The question's wording explicitly states the required scale - parse it carefully and convert accordingly. If the question asks for "thousand", divide by 1000. If it asks for "million", divide by 1,000,000. Match the question's scale exactly. IMPORTANT: When the question asks for a quantity in a specific scale, output ONLY the number representing the count of that scale unit, not the scale unit phrase itself. The question's phrasing asks for the number of scale units, not the units themselves. Parse the question carefully to determine what format is being requested.
3. Match the question's format exactly: if it asks for a number, output only that number; if it asks for a name, output only that name; if it asks for units in a specific form, match that form exactly
4. Pay careful attention to the question's wording about units, scales, and format requirements - these are correctness issues, not just style
5. Output ONLY what directly answers the question - nothing else
6. Remove ALL formatting: no markdown (**, *, _, etc.), no HTML tags, no special formatting characters
7. Remove ALL explanatory text, citations, sources, and context that doesn't directly answer the question - NO parenthetical explanations, NO lists of items, NO justifications
8. Strip away ALL extra words - the answer should be the shortest possible correct response
9. UNVERIFIED CONCLUSIONS: If the critique states the draft's conclusion is "unverified", "unsupported", or "not proven", you MUST output the EXACT answer the draft concluded. For example, if the draft concluded "1" or "Ball 1", output "1". If the draft concluded "3", output "3". Do NOT attempt to complete the analysis or output a different answer - simply output the draft's exact conclusion.

Guidelines:
- The critique's "Substance" section identifies correctness problems - these are errors that must be corrected
- The critique's "Excess" section identifies verbosity - these can be removed after correctness is fixed
- If the critique says something is wrong or incorrect, you must fix it, not just make it shorter
- If the critique states the draft's conclusion is incorrect, you must re-analyze the original question to determine the correct answer. Do not guess or output an arbitrary alternative - carefully reason through the question to find the correct solution.
- If the critique states the draft's conclusion is "unverified", "unsupported", or "not proven", see CRITICAL RULE #9 above. The draft's conclusion is the answer to output, even if not fully proven. Only attempt to complete the missing analysis if it's a trivial extension (e.g., computing one more value using the exact same method already shown), and you are absolutely certain you can do it correctly. In all other cases, output the draft's exact conclusion.
- When the draft answer is wrong, treat this as a critical correctness issue that requires you to solve the problem correctly from scratch, using the critique's feedback about what was wrong.
- If the question asks for a count/number, output ONLY that number with no additional text
- If the question asks for a name, output ONLY that name
- If the question asks for a value with units, output ONLY that value and unit (matching the question's scale)
- Preserve all information necessary for correctness
- Remove explanations unless the question explicitly asks for them
- Prefer precise nouns and verbs over multi-clause explanations
- Combine sentences wherever possible without reducing clarity
- Aim for the shortest answer that would still receive full credit from a careful grader
- If the question can be answered with a single value, name, or sentence, do so
- Do not mention the draft, the critique, or the revision process
- The final answer should read as if it were written once, cleanly and confidently
- CORRECT first, then SUCCINCT - never sacrifice correctness for brevity
- NO parentheticals, NO lists, NO explanations - just the direct answer""",
        }

        # Load dataset eagerly to avoid progress bars during episode execution
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load the dataset eagerly during initialization."""
        token = os.getenv(self.cfg.hf_token_env)
        # Suppress progress bars from Hugging Face during dataset loading
        # by redirecting stderr (where tqdm writes) to devnull
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            # snapshot_download is recommended on the dataset card
            self._data_dir = snapshot_download(
                repo_id=self.cfg.repo_id,
                repo_type="dataset",
                token=token,
            )
            self._ds = load_dataset(self._data_dir, self.cfg.config_name, split=self.cfg.split)

    def _ensure_loaded(self) -> None:
        """Ensure dataset is loaded (no-op since we load eagerly in __init__)."""
        if self._ds is None:
            self._load_dataset()

    def reset(self, *, seed: Optional[int] = None) -> dict[str, Any]:
        """
        Reset environment and return initial observation.

        Args:
            seed: Optional random seed (not used for sequential iteration)

        Returns:
            Dict of dicts for multi-agent mode: {"draft": {...}, "reflection": {...}, "revision": {...}}
            or single dict for single-agent mode: {"messages": [...], "tools": [...]}
        """
        # Reset metric counters and agent management
        super().reset(seed=seed)

        # Reset draft/reflection strings
        self._draft_string = None
        self._reflection_string = None

        # Dataset should already be loaded from __init__, but ensure it's loaded just in case
        self._ensure_loaded()
        # Simple sequential iteration; you can randomize with seed if you want.
        self._idx = (self._idx + 1) % len(self._ds)
        self._current = self._ds[self._idx]

        question = self._current.get("Question", "")
        self._original_question = question
        attachment_abs = None
        file_path = self._current.get("file_path")
        if file_path and self._data_dir:
            attachment_abs = os.path.join(self._data_dir, file_path)

        if self.use_reflection:
            # Multi-agent mode: return dict of dicts
            # Build draft agent state
            draft_system_message = {
                "role": "system",
                "content": self._agent_system_prompts["draft"],
            }
            draft_user_content = f"Question:\n{question}"
            if attachment_abs:
                draft_user_content += f"\n\nNote: There is an attachment at {attachment_abs}\nYou may need to reference this file to answer the question."
            draft_user_content += "\n\nProduce a draft answer with a brief justification."
            draft_user_message = {
                "role": "user",
                "content": draft_user_content,
            }

            # Build reflection agent state (system message only, user message added later)
            reflection_system_message = {
                "role": "system",
                "content": self._agent_system_prompts["reflection"],
            }

            # Build decision agent state (system message only, user message added later)
            decision_system_message = {
                "role": "system",
                "content": self._agent_system_prompts["decision"],
            }

            # Build revision agent state (system message only, user message added later)
            revision_system_message = {
                "role": "system",
                "content": self._agent_system_prompts["revision"],
            }

            # Initialize agent states
            self._agent_states = {
                "draft": {
                    "messages": [draft_system_message, draft_user_message],
                    "tools": self._get_tool_schemas(),
                },
                "reflection": {
                    "messages": [reflection_system_message],
                    "tools": [],
                },
                "decision": {
                    "messages": [decision_system_message],
                    "tools": [],
                },
                "revision": {
                    "messages": [revision_system_message],
                    "tools": [],
                },
            }

            # Reset retry count for new episode
            self._draft_retry_count = 0

            return {
                "draft": self._agent_states["draft"],
                "reflection": self._agent_states["reflection"],
                "decision": self._agent_states["decision"],
                "revision": self._agent_states["revision"],
            }
        else:
            # Single-agent mode (backward compatibility)
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that answers questions accurately. Use the web_search tool when you need to look up information to answer questions. When you have enough information, provide a concise final answer with just the answer (e.g., a number or short phrase).",
            }

            # Build user message content
            content = f"Question: {question}\n\n"
            if attachment_abs:
                content += f"Note: There is an attachment at {attachment_abs}\n"
                content += "You may need to reference this file to answer the question.\n\n"
            content += "Please answer this question accurately. Use the web_search tool if you need to look up information."

            user_message = {
                "role": "user",
                "content": content,
            }

            return {"messages": [system_message, user_message], "tools": self._get_tool_schemas()}

    def compute_reward(self, predicted_answer: str, reference_answer: Optional[str]) -> float:
        """
        Compute reward for a final answer.

        Args:
            predicted_answer: The agent's answer
            reference_answer: The correct answer (may be None)

        Returns:
            Reward value (1.0 if exact match, else 0.0)
        """
        if not reference_answer:
            return 0.0
        return 1.0 if predicted_answer == reference_answer else 0.0

    def last(self) -> tuple[Any, float, bool, dict[str, Any]]:
        """
        Return dict of dicts with one dict per agent.

        Returns:
            Tuple of (observation_dict, reward, done, info)
        """
        if self.use_reflection:
            obs_dict = {
                "draft": self._agent_states.get("draft", {"messages": [], "tools": []}),
                "reflection": self._agent_states.get("reflection", {"messages": [], "tools": []}),
                "decision": self._agent_states.get("decision", {"messages": [], "tools": []}),
                "revision": self._agent_states.get("revision", {"messages": [], "tools": []}),
            }
            return obs_dict, 0.0, False, {}
        else:
            # Single-agent mode: return single dict
            return super().last()

    def step(self, action: Any, agent_id: Optional[str] = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dict with "messages" key containing list with assistant message
            agent_id: Agent ID that produced the action (if None, uses _agent_selection)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        assert self._current is not None, "Call reset() first"

        # Use agent_id if provided, otherwise use _agent_selection
        current_agent_id = agent_id if agent_id is not None else self._agent_selection

        info: dict[str, Any] = {"task_id": self._current.get("task_id")}

        # Store reference answer in info for reward computation
        ref = (self._current.get("Final answer") or "").strip()
        if ref:
            info["reference_answer"] = ref

        if self.use_reflection:
            # Multi-agent mode: handle agent cycling
            import json

            # Increment step count (parent class does this for single-agent mode)
            self._step_count += 1

            # Extract assistant message from action
            messages = action["messages"]
            assistant_message = None
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_message = msg
                    break

            if assistant_message is None:
                raise ValueError("Action must contain an assistant message")

            # Handle tool calls
            if assistant_message.get("tool_calls"):
                # Execute tools and replace agent's message list with tool messages
                tool_calls = assistant_message["tool_calls"]
                self._tool_calls_count += len(tool_calls)
                tool_messages = []

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    tool_result = self.execute_tool(function_name, function_args)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result),
                    }
                    tool_messages.append(tool_message)

                # Replace agent's message list with tool messages
                if current_agent_id in self._agent_states:
                    self._agent_states[current_agent_id]["messages"] = tool_messages

                info["steps_taken"] = self._step_count
                info["tool_calls_count"] = self._tool_calls_count

                obs = {
                    "draft": self._agent_states["draft"],
                    "reflection": self._agent_states["reflection"],
                    "decision": self._agent_states["decision"],
                    "revision": self._agent_states["revision"],
                }
                return obs, 0.0, False, info

            # Handle final answer
            if assistant_message.get("content"):
                content = (assistant_message.get("content") or "").strip()

                if current_agent_id == "draft":
                    # Draft agent produced final answer
                    self._draft_string = content
                    # Add user message to reflection agent
                    reflection_user_message = {
                        "role": "user",
                        "content": f"Question:\n{self._original_question}\n\nDraft Answer:\n{content}\n\nProvide a structured critique focusing on correctness and opportunities to shorten the answer without loss of information.",
                    }
                    self._agent_states["reflection"]["messages"].append(reflection_user_message)
                    # Advance to reflection agent
                    self._agent_idx += 1
                    self._agent_selection = "reflection"

                    info["steps_taken"] = self._step_count
                    info["tool_calls_count"] = self._tool_calls_count

                    obs = {
                        "draft": self._agent_states["draft"],
                        "reflection": self._agent_states["reflection"],
                        "decision": self._agent_states["decision"],
                        "revision": self._agent_states["revision"],
                    }
                    return obs, 0.0, False, info

                elif current_agent_id == "reflection":
                    # Reflection agent produced final answer
                    self._reflection_string = content
                    # Add user message to decision agent
                    decision_user_message = {
                        "role": "user",
                        "content": f"Question:\n{self._original_question}\n\nDraft Answer:\n{self._draft_string}\n\nReflection Critique:\n{content}\n\nShould the draft agent retry? Answer only \"retry\" or \"proceed_to_revision\".",
                    }
                    self._agent_states["decision"]["messages"].append(decision_user_message)
                    # Advance to decision agent
                    self._agent_idx += 1
                    self._agent_selection = "decision"

                    info["steps_taken"] = self._step_count
                    info["tool_calls_count"] = self._tool_calls_count

                    obs = {
                        "draft": self._agent_states["draft"],
                        "reflection": self._agent_states["reflection"],
                        "decision": self._agent_states["decision"],
                        "revision": self._agent_states["revision"],
                    }
                    return obs, 0.0, False, info

                elif current_agent_id == "decision":
                    # Decision agent produced decision
                    decision = content.strip().lower()
                    # Parse decision (should be "retry" or "proceed_to_revision")
                    if decision == "retry" and self._draft_retry_count < self.max_draft_retries:
                        # Increment retry count
                        self._draft_retry_count += 1
                        # Build user message for draft agent with reflection feedback
                        draft_retry_user_message = {
                            "role": "user",
                            "content": f"Question:\n{self._original_question}\n\nPrevious Draft Answer:\n{self._draft_string}\n\nReviewer Critique:\n{self._reflection_string}\n\nCRITICAL: You MUST provide a specific answer to the question. Do not refuse to answer or say the problem is underspecified - commit to an interpretation and provide an answer. If the critique says you 'do not answer the question' or 'fail to provide the required answer format', you must commit to a specific interpretation (even if ambiguous) and provide a specific answer based on that interpretation. If the problem has multiple interpretations, provide conditional answers with specific values for each interpretation. Always end with a concrete answer that matches the question's required format.\n\nThe reviewer has identified critical correctness issues with your previous draft. The critique's \"Substance\" section lists specific errors that must be fixed. You must:\n\n1. Carefully read each correctness issue identified in the critique\n2. Systematically address each issue - do not skip any\n3. Re-solve the problem from scratch if needed, using the critique's feedback about what was wrong\n4. Ensure your new draft answer is correct, not just different\n\nCRITICAL GUIDANCE FOR ADDRESSING SPECIFIC ISSUES:\n\n- If the critique says your conclusion is \"unverified\", \"unsupported\", or \"not proven\":\n  * For optimization problems (e.g., \"which X maximizes Y\"): You MUST compute the value of Y for ALL candidate X values explicitly. Do not rely on symmetry or intuition - compute each value.\n  * For mathematical/probability problems: You MUST either provide a rigorous, complete proof OR use computational methods (explicit calculation, simulation, dynamic programming) to compute the answer. If analytical methods fail, use computational methods.\n  * For problems requiring verification: You MUST independently verify your conclusion is correct, not just that your reasoning seems plausible.\n\n- If the critique says your argument is \"invalid\", has \"gaps\", or is \"incomplete\":\n  * Abandon the failed approach entirely. Do not try to patch it - it is fundamentally flawed.\n  * Try a fundamentally different method. If you used symmetry arguments, try computational methods. If you used intuition, try explicit calculation.\n  * For optimization problems: Compute all candidate values explicitly rather than using symmetry arguments.\n  * For probability problems: Compute probabilities explicitly rather than relying on intuitive symmetry.\n\n- If the critique says you need to \"compute all values\" or \"prove rigorously\":\n  * Do exactly that - compute all candidate values explicitly or provide a complete, rigorous proof.\n  * Do not skip steps or assume symmetry - show your work completely.\n\n- Actionable steps when analytical methods fail:\n  * Use computational methods: explicit calculation, simulation, dynamic programming, enumeration\n  * For optimization: compute the objective function for all candidates explicitly\n  * For probability: compute probabilities explicitly using the definition or simulation\n  * For mathematical proofs: provide a complete, rigorous proof with no gaps\n\nVERIFICATION AND INTERPRETATION CHECKING (CRITICAL):\n\n- When the critique says your conclusion is \"unverified\", \"unsupported\", or \"not proven\", you MUST:\n  * Check alternative problem interpretations: If the problem statement has ambiguous wording, explicitly test your conclusion under different interpretations. Compute the answer for each interpretation and verify which one is correct.\n  * Verify edge cases: Test your computation with small, simple cases where you can verify the answer manually (e.g., if computing probabilities for 100 items, first verify with 3-5 items where you can enumerate all possibilities).\n  * Use a different method to cross-check: If you used dynamic programming, also try simulation or explicit enumeration for small cases. If you used analytical methods, also try computational methods. The answers should match.\n  * Verify logical consistency: Check that your computed values make sense (e.g., probabilities should be between 0 and 1, should sum correctly if applicable, should be non-negative, etc.). If your answer seems counterintuitive, double-check your interpretation and computation.\n\n- Interpretation checking guidance:\n  * If your computation gives a different answer than expected or the critique suggests your interpretation is wrong, explicitly consider alternative interpretations of ambiguous problem statements.\n  * Test your conclusion under different interpretations to see which one is correct. If the problem has ambiguous terms or phrases, try all reasonable interpretations and compute the answer for each.\n  * If the critique questions your interpretation, you must either: (a) prove your interpretation is correct from the problem text, or (b) test alternative interpretations and use the one that gives a verifiable correct answer.\n\n- Cross-validation steps:\n  * After computing probabilities or values, verify them using a different method (e.g., if using DP, also simulate or enumerate for small cases).\n  * Check that your answer makes logical sense: probabilities should be between 0 and 1, values should be in the expected range, sums should be correct, etc.\n  * If your answer seems counterintuitive or doesn't match what you'd expect, this is a red flag - re-check your interpretation and computation.\n\nProduce a corrected draft answer that definitively addresses all correctness issues identified in the critique. Your answer must be CORRECT, not just different from the previous attempt.",
                        }
                        # Reset draft agent's state (clear previous messages, keep system prompt)
                        draft_system_message = {
                            "role": "system",
                            "content": self._agent_system_prompts["draft"],
                        }
                        self._agent_states["draft"]["messages"] = [draft_system_message, draft_retry_user_message]
                        # Reset draft string (will be updated on next draft)
                        self._draft_string = None
                        # Reset reflection string (will be updated on next reflection)
                        self._reflection_string = None
                        # Clear reflection and decision agent messages (they'll be rebuilt)
                        reflection_system_message = {
                            "role": "system",
                            "content": self._agent_system_prompts["reflection"],
                        }
                        self._agent_states["reflection"]["messages"] = [reflection_system_message]
                        decision_system_message = {
                            "role": "system",
                            "content": self._agent_system_prompts["decision"],
                        }
                        self._agent_states["decision"]["messages"] = [decision_system_message]
                        # Go back to draft agent
                        self._agent_idx = 0
                        self._agent_selection = "draft"

                        info["steps_taken"] = self._step_count
                        info["tool_calls_count"] = self._tool_calls_count

                        obs = {
                            "draft": self._agent_states["draft"],
                            "reflection": self._agent_states["reflection"],
                            "decision": self._agent_states["decision"],
                            "revision": self._agent_states["revision"],
                        }
                        return obs, 0.0, False, info
                    else:
                        # Decision is "proceed_to_revision" or max retries reached
                        # Add user message to revision agent
                        revision_user_message = {
                            "role": "user",
                            "content": f"Question:\n{self._original_question}\n\nDraft Answer:\n{self._draft_string}\n\nReviewer Critique:\n{self._reflection_string}\n\nProduce the final revised answer. FIRST fix all correctness issues identified in the critique's \"Substance\" section (including any unit/scale conversion requirements - if the question asks for a quantity in a specific scale like \"thousand\" or \"million\", convert your computed value to that scale by dividing by the appropriate factor). THEN make it succinct by removing excess identified in the \"Excess\" section. Output ONLY what directly answers the question with no formatting, no explanations, no parentheticals, no lists - just the direct answer matching the question's exact format and scale requirements.",
                        }
                        self._agent_states["revision"]["messages"].append(revision_user_message)
                        # Advance to revision agent
                        self._agent_idx += 1
                        self._agent_selection = "revision"

                        info["steps_taken"] = self._step_count
                        info["tool_calls_count"] = self._tool_calls_count

                        obs = {
                            "draft": self._agent_states["draft"],
                            "reflection": self._agent_states["reflection"],
                            "decision": self._agent_states["decision"],
                            "revision": self._agent_states["revision"],
                        }
                        return obs, 0.0, False, info

                elif current_agent_id == "revision":
                    # Revision agent produced final answer - episode done
                    info["predicted_answer"] = content
                    reward = self.compute_reward(content, ref)
                    if ref:
                        info["reference_answer"] = ref

                    info["steps_taken"] = self._step_count
                    info["tool_calls_count"] = self._tool_calls_count

                    # Add exact_match metric
                    if ref and content:
                        info["exact_match"] = 1.0 if content == ref else 0.0
                    else:
                        info["exact_match"] = 0.0

                    # Mark episode as done
                    self._agent_selection = None

                    obs = {
                        "draft": self._agent_states["draft"],
                        "reflection": self._agent_states["reflection"],
                        "decision": self._agent_states["decision"],
                        "revision": self._agent_states["revision"],
                    }
                    return obs, reward, True, info

            raise ValueError(f"Assistant message must have either tool_calls or content: {assistant_message}")

        else:
            # Single-agent mode: call parent step method
            obs, reward, done, info = super().step(action, agent_id=agent_id, info=info)

            # Add exact_match metric when done
            if done:
                pred = info.get("predicted_answer", "").strip()
                ref = info.get("reference_answer", "").strip()
                if ref and pred:
                    info["exact_match"] = 1.0 if pred == ref else 0.0
                else:
                    info["exact_match"] = 0.0

            return obs, reward, done, info
