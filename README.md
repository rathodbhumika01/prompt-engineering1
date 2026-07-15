# prompt-engineering1
# Prompt Engineering : Temperature Demo

a Python program that generates text using GPT-2 at different "temperature" values, to observe how temperature affects the randomness/creativity of a language model's output.

## What it does
- Loads GPT-2 locally using Hugging Face's `transformers` library
- Feeds the same prompt (a deadline-extension email) to the model at four temperature values: 0.2, 0.7, 1.0, 1.5
- Prints the generated response for each temperature so they can be compared side by side

## Tech used
- Python
- Hugging Face `transformers` (`pipeline`, GPT-2 model)

## How to run
bash
pip install transformers
python temperature_demo_local_B.py


## Observation
As temperature increases, the output becomes less predictable.
0.2 : stays closest to the topic, most focused/coherent.
0.7–1.0 : starts adding unrelated details but stays fluent.
1.5 : most random, trails into unrelated/incomplete ideas.

This demonstrates how temperature controls the balance between predictable (low) and creative/random (high) output in text generation.
