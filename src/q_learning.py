from typing import List


def q_value(start_value: float,
            learning_rate: float,
            reward: float,
            y: float,
            end_state: float) -> float:
    return start_value + learning_rate * (reward + y * end_state - start_value)


def q_update(table: List[List[float]],
             start_state: int,
             end_state: int,
             action: int,
             reward: float,
             learning_rate: float,
             y: float) -> float:
    """Calculates Q learning update based on the state table, chosen
    starting state and action, reward, learning rate and discount factors"""
    assert start_state < len(table)
    assert end_state < len(table)
    assert action < len(table[start_state])
    s1 = table[start_state][action]
    utilities = map(lambda end: q_value(s1, learning_rate, reward, y, end), table[end_state])
    return max(utilities)
