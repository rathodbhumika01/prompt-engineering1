# Assignment 2 - Prompt Engineering
# Generate responses to the prompt using different temperature values

from transformers import pipeline, set_seed

MODEL_NAME = "gpt2"

prompt = (
    "Subject: Request for Deadline Extension\n\n"
    "Dear Professor,\n\n"
    "I am writing to request an extension on the upcoming assignment "
    "deadline because"
)

temperatures = [0.2, 0.7, 1.0, 1.5]
max_new_tokens = 50

generator = pipeline("text-generation", model=MODEL_NAME)

print("Prompt:")
print(prompt)
print("=" * 70)

for temp in temperatures:
    set_seed(42)
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        pad_token_id=50256,
    )
    response = output[0]["generated_text"]

    print(f"Temperature = {temp}")
    print(response)
    print("-" * 70)



# Output from running the program:
# Temperature = 0.2
# Subject: Request for Deadline Extension
#
# Dear Professor,
#
# I am writing to request an extension on the upcoming assignment
# deadline because I have been told that you are not allowed to work in
# this field. Please let me know if there is any other way of
# contacting us about your position and we will consider it as a
# possibility!
#
# (Please note that my email address does NOT
# -
#
# Temperature = 0.7
# Subject: Request for Deadline Extension
#
# Dear Professor,
#
# I am writing to request an extension on the upcoming assignment
# deadline because I would like you not only in your department but
# also at least one other professor and a faculty member. The
# University of Michigan has been working with us since January 2016 as
# part that process which involves our academic community's support
# staff including members from all levels
#
# Temperature = 1.0
# Subject: Request for Deadline Extension
#
# Dear Professor,
#
# I am writing to request an extension on the upcoming assignment
# deadline because I would like a chance at getting your resume
# translated into English. Given that my name is not listed in this
# publication's "Request", what should you do about it? Your CV has
# been removed from our website and replaced with two documents
# describing his PhD
# 
#
# Temperature = 1.5
# Subject: Request for Deadline Extension
#
# Dear Professor,

# I am writing to request an extension on the upcoming assignment
# deadline because I would like a chance at some form of feedback
# during this process. Your position may be approved and it will get us
# over 3 weeks into your project so please bear in mind that we aren't
# allowed time when you finish as far down towards his/


# Observation:
# As temperature goes up, the answers get less predictable.
# At 0.2 the model stays closest to the actual topic (requesting
# contact about a position). At 0.7-1.0 it starts adding random
# unrelated details (university names, resumes, PhDs) but the
# sentences still sound fluent. At 1.5 it gets the most random,
# trailing off into unrelated/incomplete ideas. This shows temperature
# controls the balance between predictable output (low) and
# creative/random output (high).
