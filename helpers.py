import os

def get_current_directory_path():
    """
    Returns the absolute path of the current working directory.
    
    Returns:
        str: The absolute path of the current working directory.
    """
    current_path = os.getcwd()
    return current_path

def save_Prompts(input_string):
    # Strip double quotes
    processed_string = input_string.replace('"', '')
    # Remove new lines and carriage returns
    processed_string = processed_string.replace('\n', '').replace('\r', '')
    # Add quotes to the beginning and end of the string
    processed_string = '"' + processed_string + '"'
    # Get current directory path
    current_path = get_current_directory_path()
    # Define file path
    file_path = os.path.join(current_path, 'models', 'prompts.txt')
    # Append the processed string to the file
    with open(file_path, 'a') as file:
        file.write(processed_string + '\n')