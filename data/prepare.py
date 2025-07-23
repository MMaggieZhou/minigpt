import json 
def processFile(file_path):
    """
    Process the file at the given path.
    
    Args:
        file_path (str): The path to the file to be processed.
    
    Returns:
        dialogs (array): Processed data from the file.
    """
    dialogs = []

    try:
        with open(file_path, 'r') as file:
            # read the file content by lines 
            data = file.read()
            # split the data into lines
            data = data.splitlines()
            print(len(data))
            for line in data: 
                if "：" not in line:
                    continue 
                parsed = line.split("：", 1)
                person = parsed[0].strip()
                if "（" in person: 
                    person = person.split("（")[0].strip()
                message = parsed[1].strip()
                dialogs.append((person, message))
    except Exception as e:
        return f"An error occurred while processing the file: {e}"
    
    return dialogs

def match(actual, expected): 
    if expected == '*':
        return True 
    return actual == expected

def filter_dialogs(dialogs, role_one, role_two):
    ret = []
    for i in range(len(dialogs) - 1):
        dialogs_one = dialogs[i]
        dialogs_two = dialogs[i + 1]
        if not match(dialogs_one[0], role_one) or not match(dialogs_two[0], role_two):
            continue
        ret.append((dialogs_one, dialogs_two))
    return ret

def select_dialogs_from_file(file_path, role_one, role_two):
    """
    Select dialogs from the file based on the specified roles.
    
    Args:
        file_path (str): The path to the file to be processed.
        role_one (str): The first role to match.
        role_two (str): The second role to match.
    
    Returns:
        list: A list of tuples containing matched dialogs.
    """
    dialogs = processFile(file_path)
    return filter_dialogs(dialogs, role_one, role_two)

def select_dialogs_from_files(file_paths, role_one, role_two):
    """
    Select dialogs from multiple files based on the specified roles.
    
    Args:
        file_paths (list): A list of file paths to be processed.
        role_one (str): The first role to match.
        role_two (str): The second role to match.
    
    Returns:
        list: A list of tuples containing matched dialogs from all files.
    """
    all_dialogs = []
    for file_path in file_paths:
        dialogs = select_dialogs_from_file(file_path, role_one, role_two)
        all_dialogs.extend(dialogs)
    return all_dialogs


def persist_dialogs_to_file(dialogs, role_one, role_two, base_dir):
    """
    Persist the selected dialogs to a file.
    
    Args:
        dialogs (list): A list of tuples containing dialogs to be saved.
        output_file (str): The path to the output file.
    """
    file_name = f"{base_dir}/{role_one}_{role_two}_dialogs.json"
    lines = []
    for dialog in dialogs: 
        processed_line = {}
        processed_line["instruction"] = dialog[0][1]
        processed_line["input"] = ""
        processed_line["output"] = dialog[1][1]
        lines.append(processed_line)
    # persist lines to a json file
    with open(file_name, 'w', encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=4)
    print(f"Dialogs saved to {file_name}")
