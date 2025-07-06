# Whitelist_GPT
Experiments with masking logits to steer transformer outputs. TLDR: make an LLM only output approved words!


# Why don't you say Bicycle? 

## file: Llama_cpp_working
Restricting the output tokens of a modern instruction tuned LLM. Note that any letters that are turned off in the radio buttons do not appear in the outputs at the beginning of words.
Any tokens not appearing on the approved vocab list will have their logits set to zero (using a mask). In other words, the probability of selecting tokens that begin with banned letters becomes zero.

![Screenshot 2025-07-06 093649](https://github.com/user-attachments/assets/15456a9a-0677-4111-8a7c-5d49906e8c9f)



## file: gpt2_restricted_vocab_v3_letterban

Same concept, but using the much smaller GPT2 model

![Screenshot 2025-07-06 100213](https://github.com/user-attachments/assets/286ed43c-8fc2-4c03-9dd6-024dd3fe18db)

Note the output is restricted to tokens beginning with the letter "A".

![Screenshot 2025-07-06 100248](https://github.com/user-attachments/assets/f52b6d8f-54c7-46b7-b0c9-7737126703d1)

## file: gpt_restricted_from_py_list_v2

This one only allows tokens that match an approved vocab list, aka whitelist.

![Screenshot 2025-07-06 101745](https://github.com/user-attachments/assets/fa31c381-4957-4221-883e-f206fda7b098)

