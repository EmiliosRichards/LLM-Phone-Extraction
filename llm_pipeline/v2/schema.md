# Phone Number Extraction Schema

> 🧠 NOTE: This schema file applies only to the **v2 LLM pipeline**.
> The v1 version uses a simpler output format (top-level categories with lists of phone numbers), and does not require a schema.

> 🛠 NOTE: This schema represents a **planned future output format** for the LLM.
> The current prompt uses a simpler format with categories as top-level keys.

## Schema Versioning

The current schema version is `2.0.0`. Version numbers follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes to the schema structure
- MINOR: New optional fields or categories
- PATCH: Documentation updates or clarifications

Future schema changes will be documented in this file with clear migration notes. Downstream systems should check for the `schema_version` field in metadata when it becomes available.

## Output Format

The LLM should return a JSON object with the following structure:

```json
{
    "phone_numbers": [
        {
            "number": "+1 (555) 123-4567",
            "category": "Sales",
            "confidence": 0.95,
            "context": "Contact our sales team at +1 (555) 123-4567 for pricing information"
        }
    ],
    "metadata": {
        "total_numbers_found": 1,
        "processing_timestamp": "2024-03-20T15:30:00Z",
        "schema_version": "2.0.0"
    }
}
```

## JSON Schema Definition

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["phone_numbers", "metadata"],
    "properties": {
        "phone_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["number", "category", "confidence", "context"],
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "The extracted phone number in E.164 format when possible"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Sales", "Support", "General", "Recruiting", "LowValue"],
                        "description": "The category of the phone number"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score between 0 and 1"
                    },
                    "context": {
                        "type": "string",
                        "maxLength": 100,
                        "description": "The surrounding text that helped identify the number's category"
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "required": ["total_numbers_found", "processing_timestamp"],
            "properties": {
                "total_numbers_found": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Count of unique phone numbers found"
                },
                "processing_timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO 8601 timestamp of when the extraction was performed"
                },
                "schema_version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$",
                    "description": "Semantic version of the schema used for validation"
                }
            }
        }
    }
}
```

### Field Descriptions

#### Phone Numbers Array
Each phone number object contains:
- `number`: The extracted phone number in E.164 format when possible
- `category`: One of the defined categories below
- `confidence`: A float between 0 and 1 indicating confidence in the extraction
- `context`: The surrounding text that helped identify the number's category

#### Metadata
- `total_numbers_found`: Count of unique phone numbers found
- `processing_timestamp`: ISO 8601 timestamp of when the extraction was performed

## Phone Number Categories

### Current Categories

1. **Sales**
   - Primary sales contact numbers
   - Sales department numbers
   - Business development contact numbers
   - Example: "Call our sales team at..."

2. **Support**
   - Customer support numbers
   - Technical support numbers
   - Help desk numbers
   - Example: "For technical support, dial..."

3. **General**
   - Main office numbers
   - General inquiry numbers
   - Reception numbers
   - Example: "For general inquiries, contact us at..."

4. **Recruiting**
   - HR department numbers
   - Recruitment contact numbers
   - Career inquiry numbers
   - Example: "To apply for positions, call..."

5. **LowValue**
   - Fax numbers
   - Automated systems
   - Non-human contact points
   - Example: "Fax your documents to..."

<details>
<summary>Future Optional Categories</summary>

The following categories may be added in future versions:

1. **Admin**
   - Administrative office numbers
   - Executive assistant numbers
   - Office management contacts

2. **IT**
   - IT department numbers
   - System administration contacts
   - Network operations numbers

3. **Press**
   - Media relations numbers
   - Public relations contacts
   - Press office numbers
</details>

## Notes

- Phone numbers should be normalized to E.164 format when possible
- Confidence scores should reflect both the number extraction and category assignment
- Context should be limited to a reasonable length (e.g., 100 characters)
- Duplicate phone numbers should be consolidated with the highest confidence category 

### Handling Non-English Content

When processing content from non-English domains (e.g., `.de`, `.fr`, `.jp`):
- Extract phone numbers in their original format first
- Attempt to normalize to E.164 format, preserving the country code
- If normalization fails, keep the original format but note this in the context
- Consider the page's language when interpreting category context
- Example: "Kontaktieren Sie unser Verkaufsteam" should still be categorized as "Sales"

### Category Assignment Guidelines

When a number's category is unclear:
1. First attempt to infer the category from surrounding context
2. If context is ambiguous or missing:
   - Assign "General" as the category
   - Set confidence to 0.3 or lower
   - Include the ambiguous context in the context field
3. Examples of unclear cases:
   - Numbers in footers without clear labels
   - Numbers in image alt text without context
   - Numbers in generic "Contact Us" sections
   - Numbers in multi-language pages with mixed context

## Validating Output Locally

You can validate the LLM output against this schema using Python's `jsonschema` library. Here's how to set it up:

1. Install the required package:
```bash
pip install jsonschema
```

2. Create a validation script (`validate_output.py`):
```python
import json
import sys
from jsonschema import validate, ValidationError

def validate_output(output_file: str, schema_file: str) -> bool:
    """
    Validate a JSON output file against the schema.
    
    Args:
        output_file: Path to the JSON output file
        schema_file: Path to the JSON schema file
    
    Returns:
        bool: True if valid, False if invalid
    """
    try:
        with open(output_file, 'r') as f:
            output = json.load(f)
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        validate(instance=output, schema=schema)
        print("✅ Output is valid!")
        return True
    except ValidationError as e:
        print("❌ Validation failed:")
        print(f"  {e.message}")
        print(f"  Path: {' -> '.join(str(p) for p in e.path)}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_output.py <output.json> <schema.json>")
        sys.exit(1)
    
    success = validate_output(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
```

3. Save the schema to a file (`schema.json`):
```bash
# Extract the schema from this markdown file
grep -A 1000 "JSON Schema Definition" schema.md | grep -B 1000 "Field Descriptions" | grep -v "^##" > schema.json
```

4. Run the validation:
```bash
python validate_output.py output.json schema.json
```

Example output for invalid data:
```
❌ Validation failed:
  'confidence' is greater than the maximum of 1
  Path: phone_numbers -> 0 -> confidence
```

You can also use this validation in your Python code:

```python
from validate_output import validate_output

# Validate a single file
is_valid = validate_output("output.json", "schema.json")

# Or validate multiple files
for output_file in ["output1.json", "output2.json"]:
    if not validate_output(output_file, "schema.json"):
        print(f"Failed to validate {output_file}") 