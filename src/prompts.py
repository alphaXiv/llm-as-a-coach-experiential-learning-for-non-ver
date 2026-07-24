RUBRIC_SYSTEM = "You are an expert evaluator designing grading rubrics."

RUBRIC_USER = """Write a concise evaluation rubric for judging responses to the user request below.
List 4-6 quality criteria specific to this request, each as a short bullet starting with '-'.
Do not evaluate any response; only produce the rubric.

User request:
{prompt}

Rubric:"""

COACH_SYSTEM = "You are an expert coach evaluating an AI assistant's response."

COACH_USER = """Evaluate the assistant's response to the user request against the rubric.

User request:
{prompt}

Rubric:
{rubric}

Assistant response:
{response}

First, write a brief analysis (under 150 words) of the response's strengths and weaknesses against the rubric.
Then give an overall score on its own line in exactly the form "Score: X/10" where X is an integer from 1 to 10.
Finally, distill your analysis into general, transferable guidance for producing better responses to similar requests. Focus on reusable strategies, not corrections specific to this one response. Write 3-5 sentences inside <experience> tags:
<experience>
...
</experience>"""

DIRECTIVES = [
    "Address every part of the request completely.",
    "Be more specific and concrete; avoid vague generalities.",
    "Improve the structure and organization of the response.",
    "Improve factual accuracy and avoid unsupported claims.",
    "Match the tone and style the request calls for.",
    "Be more concise; remove filler and repetition.",
    "Add more depth, detail, or examples.",
    "Improve clarity and readability.",
    "Be more creative and original.",
    "The response is already strong; maintain this level of quality.",
]

DIRECTIVE_USER = """Based on the following analysis of an assistant's response, choose the single most relevant improvement directive from this numbered list. Reply with ONLY the number.

Analysis:
{analysis}

Directives:
{directives}

Number:"""

TEACHER_GUIDANCE_SYSTEM = """You are a helpful assistant. You have access to the following guidance distilled from experience on similar tasks:

{context}

Apply this guidance implicitly when answering. Never mention or quote the guidance itself."""

PAIRWISE_SYSTEM = "You are an impartial judge comparing two AI assistant responses."

PAIRWISE_USER = """User request:
{prompt}

Response A:
{a}

Response B:
{b}

Which response better addresses the user request overall, considering helpfulness, accuracy, completeness, organization, and style? Answer with exactly one word: A, B, or TIE.

Verdict:"""

SCORE_SYSTEM = "You are an impartial judge scoring an AI assistant's response."

SCORE_USER = """Score the assistant's response to the user request against the rubric.

User request:
{prompt}

Rubric:
{rubric}

Assistant response:
{response}

Reply with exactly one line of the form "Score: X/10" where X is an integer from 1 to 10. No other text.

Score:"""

GENERIC_RUBRIC = """- Fully addresses the user's request and its constraints
- Accurate and free of unsupported or fabricated claims
- Well organized and easy to follow
- Appropriate tone, style, and level of detail for the request
- Clear, fluent, engaging writing"""
