import typing

def patched_remove_dups(parameters):
    all_params = []
    for p in parameters:
        if p not in all_params:
            all_params.append(p)
    return tuple(all_params)

typing._remove_dups_flatten = patched_remove_dups

print("Patch applied. Importing...")
import function
import flask_server
print("SUCCESS!")
