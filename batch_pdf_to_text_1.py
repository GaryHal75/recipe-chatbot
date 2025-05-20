import pdfplumber
import os
import re
import pandas as pd

# Define input and output paths
INPUT_FOLDER = "Inputs"
OUTPUT_FOLDER = "Outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "flattened"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "structured"), exist_ok=True)

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

# Loop through all PDFs in the input folder
pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]

if not pdf_files:
    print("No PDFs found in the 'Inputs' folder.")

for filename in pdf_files:
    pdf_path = os.path.join(INPUT_FOLDER, filename)
    output_txt_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
    output_json_path = os.path.join(OUTPUT_FOLDER, "structured", f"{os.path.splitext(filename)[0]}.json")
    output_flattened_path = os.path.join(OUTPUT_FOLDER, "flattened", f"{os.path.splitext(filename)[0]}.txt")

    structured_data = []
    recipe = {
        "title": "",
        "servings": "",
        "cost": "",
        "image_path": "",
        "ingredients": [],
        "directions": [],
        "nutrition": {},
        "myplate": {},
        "source": filename
    }
    formatted_tables = {}  # Dictionary to store extracted tables with their locations

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

                # On first page, extract title, servings, cost, image_path
                if i == 0:
                    # Extract title from largest header line (heuristic: longest line with uppercase words)
                    possible_titles = [line.strip() for line in lines if line.strip() and sum(1 for c in line if c.isupper()) > len(line)/2]
                    if possible_titles:
                        recipe["title"] = possible_titles[0]
                    else:
                        # fallback: first non-empty line
                        for line in lines:
                            if line.strip():
                                recipe["title"] = line.strip()
                                break

                    # Extract servings and cost by scanning lines for "Makes" and "Total Cost"
                    for line in lines:
                        servings_match = re.search(r"Makes:\s*(\d+)\s*Servings?", line, re.IGNORECASE)
                        if servings_match:
                            recipe["servings"] = servings_match.group(1)
                        # Flexible cost extraction: match 1–4 literal $ symbols, possibly with spaces
                        cost_match = re.search(r"Total Cost:\s*([$\s]{1,4})", line, re.IGNORECASE)
                        if cost_match:
                            recipe["cost"] = cost_match.group(1).strip()

                    # Image path placeholder (skip actual image saving)
                    recipe["image_path"] = ""

                # Collect all table text from stored tables
                if i in formatted_tables:
                    for table in formatted_tables[i]:
                        table_texts.update(table.splitlines())

                # Parse sections: Ingredients, Directions, Nutrition Information, MyPlate Food Groups
                current_section = None
                for line in lines:
                    line_strip = line.strip()
                    if not line_strip or line_strip in table_texts:
                        continue

                    # Detect section headers
                    if re.match(r"Ingredients?", line_strip, re.IGNORECASE):
                        current_section = "ingredients"
                        continue
                    elif re.match(r"Directions?", line_strip, re.IGNORECASE):
                        current_section = "directions"
                        continue
                    elif re.match(r"Nutrition Information", line_strip, re.IGNORECASE):
                        current_section = "nutrition"
                        continue
                    elif re.match(r"MyPlate Food Groups", line_strip, re.IGNORECASE):
                        current_section = "myplate"
                        continue
                    elif re.match(r"Source", line_strip, re.IGNORECASE):
                        current_section = None
                        continue

                    # Parse content by section
                    if current_section == "ingredients":
                        # Stop if next section header is found
                        if re.match(r"Directions?", line_strip, re.IGNORECASE):
                            current_section = "directions"
                            continue
                        # Accept bullet lines (starting with -, *, •, or digits + dot)
                        if re.match(r"^[-\*\u2022]\s*(.+)", line_strip):
                            ingredient = re.sub(r"^[-\*\u2022]\s*", "", line_strip)
                            recipe["ingredients"].append(ingredient)
                        elif line_strip and not re.match(r"^\d+\. ", line_strip):
                            # Also accept lines that are not numbered steps as ingredients
                            recipe["ingredients"].append(line_strip)
                    elif current_section == "directions":
                        # Accept numbered steps
                        step_match = re.match(r"^\d+\.\s*(.+)", line_strip)
                        if step_match:
                            recipe["directions"].append(step_match.group(1))
                        elif line_strip:
                            # Also accept lines without numbering if no steps yet
                            if not recipe["directions"]:
                                recipe["directions"].append(line_strip)
                            else:
                                # Possibly continuation of last step
                                recipe["directions"][-1] += " " + line_strip
                    elif current_section == "nutrition":
                        # Parse lines like Nutrient: Value
                        nut_match = re.match(r"^([^:]+):\s*(.+)$", line_strip)
                        if nut_match:
                            key = nut_match.group(1).strip()
                            val = nut_match.group(2).strip()
                            recipe["nutrition"][key] = val
                    elif current_section == "myplate":
                        # Parse pairs like Group: Value
                        mp_match = re.match(r"^([^:]+):\s*(.+)$", line_strip)
                        if mp_match:
                            key = mp_match.group(1).strip()
                            val = mp_match.group(2).strip()
                            recipe["myplate"][key] = val

                for j in range(len(lines)):
                    if lines[j] not in table_texts:
                        structured_data.append(lines[j])

            # Insert Table Placeholder with Clear Formatting
            if i in formatted_tables:
                structured_data.append(f"\n=== PAGE {i+1} - TABLE(S) ===\n")
                structured_data.append("\n\n".join(formatted_tables[i]))  # Add extra spacing for clarity
                structured_data.append("=" * 40)  # Table separator

    # Save structured text output
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(structured_data))

    print(f"✅ Extraction complete for {filename}! Check output: {output_txt_path}")

    # Save structured JSON
    import json
    with open(output_json_path, "w") as jf:
        json.dump(recipe, jf, indent=2)

    # Save flattened version for FAISS
    with open(output_flattened_path, "w") as tf:
        tf.write(f"TITLE: {recipe.get('title', '')}\n")
        tf.write(f"SERVINGS: {recipe.get('servings', '')}\n")
        tf.write(f"COST: {recipe.get('cost', '')}\n\n")
        tf.write("INGREDIENTS:\n" + "\n".join(recipe.get("ingredients", [])) + "\n\n")
        tf.write("DIRECTIONS:\n" + "\n".join(recipe.get("directions", [])) + "\n\n")
        tf.write("NUTRITION:\n" + "\n".join([f"{k}: {v}" for k, v in recipe.get("nutrition", {}).items()]) + "\n\n")
        tf.write("FOOD GROUPS:\n" + "\n".join([f"{k}: {v}" for k, v in recipe.get("myplate", {}).items()]) + "\n\n")
        tf.write(f"SOURCE: {recipe.get('source', '')}\n")