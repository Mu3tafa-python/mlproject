
import os
import dill


def save_object(file_path, obj):
    # Create the parent directory if it does not exist
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the object using dill
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)
       
                   