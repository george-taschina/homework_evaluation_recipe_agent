"""Q7 (bonus): generate a synthetic evaluation dataset with structured output.

    uv run python gen_synthetic.py

For each of the 15 recipes it asks gpt-4o-mini for 5 questions (1 factual,
1 how-to, 1 detail, 1 substitution, 1 comparison) using the Pydantic models
from the homework, then saves them to synthetic_scenarios.csv.
"""

import csv
import json

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from pydantic_ai import Agent

from recipe_tools import RECIPES


class Question(BaseModel):
    question: str
    category: str
    type: str


class GeneratedQuestions(BaseModel):
    questions: list[Question]


generation_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=GeneratedQuestions,
    instructions=(
        "You generate evaluation questions for a recipe assistant that answers "
        "from a fixed collection of recipes. Generate exactly 5 questions: "
        "1 factual (ingredients), 1 how-to, 1 detail (temperature/time), "
        "1 substitution, 1 comparison with another dish. Give each question a "
        "short category and a type."
    ),
)

generation_prompt = """Given this recipe, generate 5 questions a user might ask.
Include: 1 factual, 1 how-to, 1 detail, 1 substitution, 1 comparison with another dish.
Recipe: {name}
Cuisine: {cuisine}
Ingredients: {ingredients}
Instructions: {instructions}"""


def main() -> None:
    rows = []

    for recipe in RECIPES:
        prompt = generation_prompt.format(
            name=recipe['name'],
            cuisine=recipe['cuisine'],
            ingredients=', '.join(recipe['ingredients']),
            instructions=recipe['instructions'],
        )
        result = generation_agent.run_sync(prompt)
        for q in result.output.questions:
            rows.append({
                'question': q.question,
                'category': q.category,
                'type': q.type,
            })
        print(f"{recipe['name']}: {len(result.output.questions)} questions "
              f"({result.usage.input_tokens} in / {result.usage.output_tokens} out tokens)")

    with open('synthetic_scenarios.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'category', 'type'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} synthetic questions to synthetic_scenarios.csv")


if __name__ == "__main__":
    main()
