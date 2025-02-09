


# functions for saving and loading state dictionaries as .pkl

import pickle


# saving function 
def save_model_parameters(object_to_save, file_path):
    print('<===== Saving =====>')
    state_dict = object_to_save.state_dict()
    with open(file_path, 'wb') as f:
        pickle.dump(state_dict, f)
        print('<===== Done =====>')
        
# loading function
def load_state_dict_from_file(file_path):
    with open(file_path, 'rb') as f:
        state_dict = pickle.load(f)
    return state_dict


