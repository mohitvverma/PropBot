import docx

def load_and_convert_docx_to_json(file_path):
    try:
        # Load the .docx file
        doc = docx.Document(file_path)

        # Extract text from the document
        full_text = "\n".join([para.text for para in doc.paragraphs])

        # Split the text into potential key-value pairs using the pipe "|" as a delimiter
        data_pairs = full_text.split(" | ")

        # Create a dictionary to hold the data
        data_dict = {}

        # Process each pair and attempt to split into key and value
        for pair in data_pairs:
            if ": " in pair:
                try:
                    key, value = pair.split(": ", 1)
                    data_dict[key.strip()] = value.strip()
                except ValueError:
                    # Handle cases where split fails or the format is not correct
                    data_dict[pair.strip()] = None  # Store None if no proper key-value pair is found
            else:
                # Handle cases where the delimiter is not found
                data_dict[pair.strip()] = None

        # Return the JSON data as a dictionary
        return data_dict

    except Exception as e:
        raise e