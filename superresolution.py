from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

rare_tokens = []

for token_id in range(49000, 49406):
    token   = tokenizer.convert_ids_to_tokens(token_id)
    decoded = tokenizer.convert_tokens_to_string([token]).strip()

    if (
        len(decoded) >= 2 and
        len(decoded) <= 6 and
        " " not in decoded and
        decoded.isalpha()
    ):
        rare_tokens.append((token_id, decoded))

print(f"Found {len(rare_tokens)} rare tokens:\n")
for tid, text in rare_tokens[:30]:
    print(f"Token ID: {tid} → '{text}'")