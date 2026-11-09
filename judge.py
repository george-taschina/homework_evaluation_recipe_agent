import json
import sys

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from typing import Literal
from pydantic_ai import Agent


class JudgeEvaluation(BaseModel):
    reasoning: str
    label: Literal["good", "bad"]


judge_instructions = """You are an expert evaluator assessing a recipe assistant that answers
cooking questions using a fixed recipe collection.

A response is "good" if:
1. It accurately answers using ONLY information from the recipe collection
2. It correctly identifies when a recipe is not available and says so
3. It correctly identifies out-of-scope questions (not about cooking/recipes) and declines

A response is "bad" if ANY of these apply:
1. It makes up recipes, ingredients, or instructions not in the collection (hallucination)
2. It provides cooking advice (substitutions, nutrition info, storage tips) that it
   invented rather than found in the recipe data
3. It answers a question it should have declined (not about recipes, or recipe not available)
4. It gives incorrect information from the recipes (wrong times, wrong ingredients, wrong steps)
5. It says it cannot help when a matching recipe exists in the collection

How to apply these rules:
- Judge only the content the response actually asserts. An offer to help further, to
  show recipes or alternatives that may exist in the collection, or to clarify is NOT
  by itself bad - evaluate the answer that was given, not hypothetical follow-ups.
- When the assistant honestly states the collection lacks the requested information
  (e.g. nutrition information or substitution rules), that is good. Suggesting how the
  user themselves could proceed, without the assistant fabricating a specific result,
  does not make the response bad.
- You do not have the recipe collection in front of you, so absence from your view of
  it is never itself evidence of fabrication. A response that answers the question
  directly with specific, internally consistent recipe details (ingredients, steps,
  times) is grounded - judge what the response asserts, not what you cannot verify.
  Do not reason that the assistant "should not have" provided details unless certain
  they exist. Mark a response bad only on evidence visible in it: fabricated or
  self-contradictory facts, invented advice asserted as the answer, wrong recipe
  details, or a failure to answer or decline.
- Stay strict about hallucination: any invented fact the response asserts as an answer
  - substitution advice for a substitution question, wrong times or ingredients, or a
  full set of cooking instructions for a recipe the assistant said it does not have -
  counts as bad. General cooking knowledge that goes beyond the recipe text counts as
  hallucination when the response asserts it as the answer.
"""

judge_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=JudgeEvaluation,
    instructions=judge_instructions,
)

input_file = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
output_file = input_file.replace('.json', '_judged.json')

with open(input_file) as f:
    results = json.load(f)

for i, row in enumerate(results):
    prompt = f"""Question: {row['question']}
Agent response: {row['output']}"""

    evaluation = judge_agent.run_sync(prompt)
    row['judge_label'] = evaluation.output.label
    row['judge_reasoning'] = evaluation.output.reasoning
    print(f"[{i+1}/{len(results)}] {row['judge_label']}: {row['question']}")

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved judged results to {output_file}")
