import re

# Paths (customize as needed)
input_file = 'Outputs/2021_amazing_athletes_fdd.issued.4.30.21.txt'
output_file = 'Outputs/cleaned_amazing-athletes_fdd_text.txt'

def clean_text_with_flags(text):
    # Replace cid tags like cid:103 with a marker
    text = re.sub(r'cid:\d+', 'REGEXED', text)

    # Replace non-ASCII characters with "REGEXED"
    #text = re.sub(r'[^\x00-\x7F]+', ' REGEXED ', text)

    # Replace form feeds with a newline
    text = re.sub(r'\x0c+', '\n', text)

    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()

# Read input
with open(input_file, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Clean it up
cleaned_text = clean_text_with_flags(raw_text)

# Preview for sanity
print("\n--- CLEANED TEXT PREVIEW (First 1000 characters) ---\n")
print(cleaned_text[:1000])

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print(f"\n✅ Cleaned text saved to: {output_file}")