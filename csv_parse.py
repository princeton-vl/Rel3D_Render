import json
import numpy as np
from pathlib import Path

DESCPT = 1
RESULT = 3


def uid_from_description(ddata, state_name):
    '''
    :param ddata: the json string describing the task of arranging the two objects
    :ret: relation - obj1_uid - obj2_uid - state_name - camera_string
    '''
    obj1 = ddata['objectInfos'][0]
    obj2 = ddata['objectInfos'][1]
    obj1_uid = obj1['cat'] + '_' + obj1['id']
    obj2_uid = obj2['cat'] + '_' + obj2['id']
    relation = ddata['relationInfo']

    return ' - '.join([relation, obj1_uid, obj2_uid, state_name])


def all_valid_tasks(task_csv):
    import csv

    def is_valid_entry(item):
        try:
            json.loads(item[RESULT])
        except json.decoder.JSONDecodeError:
            return False

        return True

    def is_not_across_relation(item):
        return json.loads(item[DESCPT])['relationInfo'] != 'across'

    with task_csv.open('r') as f:
        reader = csv.reader(f)
        tasks = list(filter(is_not_across_relation, filter(is_valid_entry, reader)))

    return tasks


def all_valid_tasks_multiple(task_csvs):
    task_csvs = [Path(x).resolve() for x in task_csvs.split(',')]
    import csv

    task_strings = []
    for task_csv in task_csvs:
        with task_csv.open('r') as f:
            for row in csv.reader(f):
                task_strings.append(row)

    # Convert to object
    tasks = []
    for string in task_strings:
        if task_strings.count(string) != 1: # Check duplicates
            print(string)
            raise RuntimeError('duplicate tasks detected')

        try: 
            if json.loads(string[DESCPT])['relationInfo'] == 'across':
                continue

            json.loads(string[RESULT])
            tasks.append(string)
        except json.decoder.JSONDecodeError:
            continue

    return tasks


def keep_final_state(result_data):
    """
    Checks whether we should keep finalState. Computes "distance" between finalState and finalState2, if the distance
    is less than a threshold, then return True. If True, both finalState and finalState2 should be rendered; else only
    finalState2 should be rendered
    Input:
        result_data
    Output:
        keep:(bool)
    """


    def get_position_vector(state):
        pos_vector = np.zeros(6)
        pos_vector[0] = state['objectInfos'][0]['position']['x']
        pos_vector[1] = state['objectInfos'][0]['position']['y']
        pos_vector[2] = state['objectInfos'][0]['position']['z']
        pos_vector[3] = state['objectInfos'][1]['position']['x']
        pos_vector[4] = state['objectInfos'][1]['position']['y']
        pos_vector[5] = state['objectInfos'][1]['position']['z']
        return pos_vector


    DISTANCE_THRES = 0.1  # found from manual inspection
    final_state_pos_vec = get_position_vector(result_data['finalState'])
    final_state2_pos_vec = get_position_vector(result_data['finalState2'])
    distance = np.linalg.norm(final_state_pos_vec - final_state2_pos_vec, 2)
    if distance < DISTANCE_THRES:
        return False
    else:
        return True


def valid_state_from_result(result_data):
    ''' Parse valid states such as 'initialState', 'finalState' and 'finalState2' from the collected result data.
    :param result_data: the parsed dictionary of AMT results
    :ret: a list of valid state names such as ['initialState', 'finalState']
    '''
    valid_states = ['initialState', 'finalState']
    try:
        if result_data['fig_2_correct'] == 'Yes': # Should append finalState2 as well
            valid_states.append('finalState2')
            # Now should check finalState
            if not keep_final_state(result_data):
                valid_states.remove('finalState')

        elif result_data['fig_2_correct'] == 'No':
            pass
        else:
            raise ValueError(
                f"Unexpected value {result_data['fig_2_correct']}")
    except KeyError:
        # breakpoint()
        print(result_data)
        raise

    assert all(state in result_data.keys() for state in valid_states)
    return valid_states


def load_row(row):
    descpt_data = json.loads(row[DESCPT])
    result_data = json.loads(row[RESULT])
    return descpt_data, result_data
