import pdfplumber
import json
import re
import pandas as pd  # Import Pandas for table processing

#Individual Text Process: Convert PDF document into clean text format to prep for text chunking
#Use this one for testing edge case pages that are problematic (e.g., tables, images, etc.)

# Paths
pdf_path = "Inputs/wahlburgers.PDF"
output_path = "Outputs/wahlburgers_extracted-full.txt"

structured_data = []
formatted_tables = {}  # Dictionary to store extracted tables with their locations

def format_table(table):
    """Formats extracted tables dynamically, merging stacked headers and reshaping if necessary."""
    if not table:
        return ""

    df = pd.DataFrame(table).fillna("")  # Convert to DataFrame and fill empty values
    
    # Detect and merge stacked headers dynamically
    num_header_rows = 0
    for idx, row in df.iterrows():
        if row.isnull().sum() > len(row) // 2:  # Detects when rows are mostly empty
            num_header_rows += 1
        else:
            break  # Stop when we reach real data

    # Ensure we have valid header rows
    if num_header_rows >= len(df):
        return df.to_string(index=False, header=True)  # If no valid headers, return as-is

    # Merge headers only if we have valid header data
    new_columns = [' '.join(filter(None, col)).strip() for col in zip(*df.iloc[:num_header_rows].values)]
    
    # Ensure new column length matches DataFrame width
    if len(new_columns) == len(df.columns):
        df.columns = new_columns
        df = df.iloc[num_header_rows:].reset_index(drop=True)

    # Reshape if necessary (detecting numeric-based category columns)
    potential_numeric_columns = [col for col in df.columns if df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).sum() > len(df) * 0.8]
    if len(potential_numeric_columns) > 1:
        df = df.melt(id_vars=[col for col in df.columns if col not in potential_numeric_columns], 
                     var_name="Dynamic Category", value_name="Value")

    return df.to_string(index=False, header=True)  # Preserve headers for readability

def is_section_heading(text, next_line):
    """Determine if a text line is a section heading."""
    # Example: If it's fully capitalized, assume it's a heading
    if text.isupper():
        return True
    # Detect common FDD section patterns like "ITEM 1: OVERVIEW"
    if re.match(r"^ITEM \d+[:\s]", text, re.IGNORECASE):
        return True
    # If the next line is empty or a separator, treat it as a heading
    if next_line.strip() == "" or re.match(r"[-=]{3,}", next_line):
        return True
    return False

# === PASS 1: Extract Tables and Store Locations ===
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        if tables:
            formatted_tables[i] = [format_table(table) for table in tables]

# === PASS 2: Extract Text and Insert Table Placeholders ===
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        structured_data.append(f"\n=== PAGE {i+1} ===\n")

        text = page.extract_text()
        if text:
            lines = text.splitlines()
            table_texts = set()

            # Collect all table text from stored tables
            if i in formatted_tables:
                for table in formatted_tables[i]:
                    table_texts.update(table.splitlines())

            for j in range(len(lines)):
                # Only add text lines that are not in table data
                if lines[j] not in table_texts:
                    structured_data.append(lines[j])
                    if j < len(lines) - 1 and is_section_heading(lines[j], lines[j + 1]):
                        structured_data.append(f"**SECTION HEADING: {lines[j]}**")

        # Insert Table Placeholder with Clear Formatting
        if i in formatted_tables:
            structured_data.append(f"\n=== PAGE {i+1} - TABLE(S) ===\n")
            structured_data.append("\n\n".join(formatted_tables[i]))  # Add extra spacing for clarity
            structured_data.append("=" * 40)  # Table separator

# Save structured text output
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(structured_data))

print(f"✅ Extraction complete! Check the structured text output: {output_path}")